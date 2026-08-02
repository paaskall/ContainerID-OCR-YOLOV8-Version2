import os
import re
import json
from pathlib import Path
from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS
import mysql.connector
from mysql.connector import pooling, Error
from datetime import datetime
from decimal import Decimal

app = Flask(__name__)
CORS(app)

DEBUG_ROOT = Path(os.environ.get("DEBUG_OCR_ROOT", "./debug_ocr_router"))

# Database Configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 3306)),
    'database': os.environ.get('DB_NAME', 'autogate_system'),
    'user': os.environ.get('DB_USER', 'admin'),
    'password': os.environ.get('DB_PASSWORD', 'passwordku123'),
    'pool_name': 'weight_pool',
    'pool_size': 10,
    'pool_reset_session': True
}

# Create connection pool
connection_pool = None
try:
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
    print("✓ Database connection pool created successfully")
except Error as e:
    print(f"✗ Database connection failed: {e}")
    print("  Weight endpoints will not work without database connection")

def get_db_connection():
    """Get database connection from pool"""
    if connection_pool is None:
        raise Exception("Database connection pool not available")
    return connection_pool.get_connection()

def decimal_to_float(data):
    """Convert Decimal objects to float for JSON serialization"""
    if isinstance(data, list):
        return [decimal_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {k: decimal_to_float(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    return data

def _parse_filename(filename: str) -> dict:
    """Parse nama file ROI menjadi dict metadata."""
    stem = Path(filename).stem

    suffix = ""
    if stem.endswith("_RAW"):
        suffix = "RAW"
        stem = stem[:-4]
    elif stem.endswith("_PRE"):
        suffix = "PRE"
        stem = stem[:-4]

    pattern = re.compile(
        r"^(?P<ts>\d+)"
        r"_cam(?P<cam>\d+)"
        r"_(?P<cls>[^_]+(?:_[^_]+)*?)"
        r"_conf(?P<conf>[\d.]+)"
        r"_(?P<engine>[^_]+(?:_[^_]+)*?)"
        r"_(?P<result>[^_]+)$"
    )
    m = pattern.match(stem)
    if not m:
        return {"filename": filename, "suffix": suffix}

    return {
        "filename": filename,
        "suffix": suffix,
        "ts_ms": int(m.group("ts")),
        "camera_id": int(m.group("cam")),
        "class_name": m.group("cls"),
        "confidence": float(m.group("conf")),
        "engine": m.group("engine"),
        "ocr_result": m.group("result"),
    }


def _read_meta_txt(txt_path: Path) -> dict:
    """Baca file .txt metadata menjadi dict."""
    meta = {}
    try:
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
    except Exception:
        pass
    return meta


def _list_image_pairs(detection_dir: Path) -> list:
    """Dari satu folder detection_type, kembalikan list pasangan RAW+PRE beserta metadata."""
    if not detection_dir.is_dir():
        return []

    txt_files  = {f.stem: f for f in detection_dir.glob("*.txt")}
    raw_files  = {f.stem.replace("_RAW", ""): f for f in detection_dir.glob("*_RAW.jpg")}
    pre_files  = {f.stem.replace("_PRE", ""): f for f in detection_dir.glob("*_PRE.jpg")}

    keys = sorted(set(raw_files) | set(pre_files), reverse=True)

    result = []
    for key in keys:
        src_file = raw_files.get(key) or pre_files.get(key)
        parsed   = _parse_filename(src_file.name)

        txt_meta = {}
        for tk, tf in txt_files.items():
            if tk.startswith(key[:10]):
                txt_meta = _read_meta_txt(tf)
                break

        result.append({
            **parsed,
            **{k: v for k, v in txt_meta.items() if k not in parsed},
            "has_raw": key in raw_files,
            "has_pre": key in pre_files,
            "key": key,
        })

    return result


# ============ OCR ENDPOINTS ============

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List all dates and their sessions"""
    if not DEBUG_ROOT.is_dir():
        return jsonify([])

    result = []
    for date_dir in sorted(DEBUG_ROOT.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue

        sessions = []
        for sess_dir in sorted(date_dir.iterdir(), reverse=True):
            if not sess_dir.is_dir() or not sess_dir.name.startswith("session_"):
                continue

            session_id      = sess_dir.name.replace("session_", "")
            detection_types = [d.name for d in sess_dir.iterdir() if d.is_dir()]
            total           = sum(
                len(list((sess_dir / dt).glob("*_RAW.jpg")))
                for dt in detection_types
                if (sess_dir / dt).is_dir()
            )

            sessions.append({
                "session_id":      session_id,
                "detection_types": sorted(detection_types),
                "total_detections": total,
            })

        if sessions:
            result.append({"date": date_dir.name, "sessions": sessions})

    return jsonify(result)


@app.route("/api/sessions/<date>/<session_id>/detections", methods=["GET"])
def list_detections(date: str, session_id: str):
    """List all detections in a session"""
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}"
    if not sess_dir.is_dir():
        abort(404, description=f"Session tidak ditemukan: {date}/session_{session_id}")

    filter_type = request.args.get("type", "").strip()

    result = []
    for det_dir in sorted(sess_dir.iterdir()):
        if not det_dir.is_dir():
            continue
        if filter_type and det_dir.name.lower() != filter_type.lower():
            continue

        pairs = _list_image_pairs(det_dir)
        for item in pairs:
            item["detection_type"] = det_dir.name
            item["date"]           = date
            item["session_id"]     = session_id

            base = f"/api/sessions/{date}/{session_id}/image/{det_dir.name}"
            item["url_raw"] = f"{base}/{item['key']}_RAW.jpg" if item["has_raw"] else None
            item["url_pre"] = f"{base}/{item['key']}_PRE.jpg" if item["has_pre"] else None

        result.extend(pairs)

    result.sort(key=lambda x: x.get("ts_ms", 0), reverse=True)
    return jsonify(result)


@app.route("/api/sessions/<date>/<session_id>/image/<detection_type>/<filename>", methods=["GET"])
def serve_image(date: str, session_id: str, detection_type: str, filename: str):
    """Serve ROI image file"""
    img_path = DEBUG_ROOT / date / f"session_{session_id}" / detection_type / filename

    if not img_path.is_file():
        abort(404, description=f"Gambar tidak ditemukan: {filename}")

    ext  = img_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return send_file(img_path, mimetype=mime)


@app.route("/api/sessions/<date>/<session_id>/summary", methods=["GET"])
def session_summary(date: str, session_id: str):
    """Get summary statistics for a session"""
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}"
    if not sess_dir.is_dir():
        abort(404, description=f"Session tidak ditemukan: {date}/session_{session_id}")

    summary = {
        "date":       date,
        "session_id": session_id,
        "by_detection_type": {},
    }

    for det_dir in sorted(sess_dir.iterdir()):
        if not det_dir.is_dir():
            continue

        pairs   = _list_image_pairs(det_dir)
        engines = {}
        results = {}

        for item in pairs:
            eng = item.get("engine", "unknown")
            res = item.get("ocr_result", "unknown")
            engines[eng] = engines.get(eng, 0) + 1
            results[res] = results.get(res, 0) + 1

        summary["by_detection_type"][det_dir.name] = {
            "total":       len(pairs),
            "engines":     engines,
            "ocr_results": results,
        }

    return jsonify(summary)


@app.route("/api/sessions/<date>/<session_id>/images/<detection_type>", methods=["GET"])
def get_images_by_detection(date, session_id, detection_type):
    """Get all images for a specific detection type"""
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}" / detection_type

    if not sess_dir.is_dir():
        abort(404, description="Detection type tidak ditemukan")

    pairs = _list_image_pairs(sess_dir)

    result = []
    for item in pairs:
        base = f"/api/sessions/{date}/{session_id}/image/{detection_type}"

        result.append({
            "filename": item.get("filename"),
            "confidence": item.get("confidence"),
            "ocr_result": item.get("ocr_result"),
            "url_raw": f"{base}/{item['key']}_RAW.jpg" if item["has_raw"] else None,
            "url_pre": f"{base}/{item['key']}_PRE.jpg" if item["has_pre"] else None,
        })

    return jsonify(result)


@app.route("/api/sessions/<date>", methods=["GET"])
def get_sessions_by_date(date):
    """Get all sessions for a specific date"""
    date_dir = DEBUG_ROOT / date

    if not date_dir.is_dir():
        abort(404, description=f"Tanggal tidak ditemukan: {date}")

    sessions = []
    for sess_dir in sorted(date_dir.iterdir(), reverse=True):
        if not sess_dir.is_dir() or not sess_dir.name.startswith("session_"):
            continue

        session_id = sess_dir.name.replace("session_", "")
        detection_types = [d.name for d in sess_dir.iterdir() if d.is_dir()]
        total = sum(
            len(list((sess_dir / dt).glob("*_RAW.jpg")))
            for dt in detection_types
            if (sess_dir / dt).is_dir()
        )

        sessions.append({
            "session_id": session_id,
            "detection_types": sorted(detection_types),
            "total_detections": total,
        })

    return jsonify({
        "date": date,
        "sessions": sessions
    })


@app.route("/api/sessions/<date>/<session_id>", methods=["GET"])
def get_session_detail(date, session_id):
    """Get detailed information about a specific session"""
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}"

    if not sess_dir.is_dir():
        abort(404, description=f"Session tidak ditemukan: {date}/session_{session_id}")

    detection_types = []
    total = 0

    for det_dir in sess_dir.iterdir():
        if not det_dir.is_dir():
            continue

        detection_types.append(det_dir.name)
        total += len(list(det_dir.glob("*_RAW.jpg")))

    return jsonify({
        "date": date,
        "session_id": session_id,
        "detection_types": sorted(detection_types),
        "total_detections": total
    })


@app.route("/api/sessions/<date>/<session_id>/detections/<detection_type>", methods=["GET"])
def get_detections_by_type(date, session_id, detection_type):
    """Get all detections for a specific detection type within a session"""
    det_dir = DEBUG_ROOT / date / f"session_{session_id}" / detection_type

    if not det_dir.is_dir():
        abort(404, description="Detection type tidak ditemukan")

    pairs = _list_image_pairs(det_dir)

    result = []
    for item in pairs:
        base = f"/api/sessions/{date}/{session_id}/image/{detection_type}"

        item["detection_type"] = detection_type
        item["date"] = date
        item["session_id"] = session_id

        item["url_raw"] = f"{base}/{item['key']}_RAW.jpg" if item["has_raw"] else None
        item["url_pre"] = f"{base}/{item['key']}_PRE.jpg" if item["has_pre"] else None

        result.append(item)

    result.sort(key=lambda x: x.get("ts_ms", 0), reverse=True)
    return jsonify(result)


@app.route("/api/sessions/<date>/<session_id>/detection-types", methods=["GET"])
def get_detection_types(date, session_id):
    """Get all detection types available in a session"""
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}"

    if not sess_dir.is_dir():
        abort(404, description=f"Session tidak ditemukan: {date}/session_{session_id}")

    detection_types = []
    for det_dir in sess_dir.iterdir():
        if det_dir.is_dir():
            raw_count = len(list(det_dir.glob("*_RAW.jpg")))
            detection_types.append({
                "name": det_dir.name,
                "count": raw_count
            })

    return jsonify({
        "date": date,
        "session_id": session_id,
        "detection_types": sorted(detection_types, key=lambda x: x["name"])
    })


# ============ WEIGHT ENDPOINTS ============

@app.route("/api/sessions/<date>/<session_id>/weight", methods=["GET"])
def get_session_weight(date: str, session_id: str):
    """
    Get weight data for a specific session
    Query params:
    - include_history: bool (default false) - include all weight history for this session
    - limit: int (default 10) - limit number of records
    """
    if connection_pool is None:
        abort(503, description="Database connection not available")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        limit = int(request.args.get('limit', 10))
        
        if include_history:
            query = """
                SELECT id, container_number, truck_id, plate_number, 
                       weight_kg, weight_timestamp, scale_id, operator_name, notes
                FROM weight_logs 
                WHERE session_id = %s 
                ORDER BY weight_timestamp DESC
                LIMIT %s
            """
            cursor.execute(query, (session_id, limit))
        else:
            query = """
                SELECT id, container_number, truck_id, plate_number, 
                       weight_kg, weight_timestamp, scale_id, operator_name, notes
                FROM weight_logs 
                WHERE session_id = %s 
                ORDER BY weight_timestamp DESC 
                LIMIT 1
            """
            cursor.execute(query, (session_id,))
        
        weights = cursor.fetchall()
        
        statistics = {}
        if len(weights) > 1:
            weight_values = [float(w['weight_kg']) for w in weights if w['weight_kg'] is not None]
            if weight_values:
                statistics = {
                    "min_weight_kg": min(weight_values),
                    "max_weight_kg": max(weight_values),
                    "avg_weight_kg": sum(weight_values) / len(weight_values),
                    "total_measurements": len(weight_values)
                }
        
        cursor.close()
        conn.close()
        
        weights = decimal_to_float(weights)
        
        response = {
            "date": date,
            "session_id": session_id,
            "has_weight_data": len(weights) > 0,
            "weights": weights,
            "statistics": statistics
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error getting weight data: {e}")
        abort(500, description=f"Database error: {str(e)}")


@app.route("/api/weight/container/<container_number>", methods=["GET"])
def get_weight_by_container(container_number: str):
    """
    Get weight data for a specific container number
    Query params:
    - limit: int (default 10) - limit number of records
    - start_date: date (optional) - filter by start date
    - end_date: date (optional) - filter by end date
    """
    if connection_pool is None:
        abort(503, description="Database connection not available")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT id, session_id, truck_id, plate_number, 
                   weight_kg, weight_timestamp, scale_id, operator_name, notes
            FROM weight_logs 
            WHERE container_number = %s
        """
        params = [container_number]
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            query += " AND weight_timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND weight_timestamp <= %s"
            params.append(end_date)
            
        query += " ORDER BY weight_timestamp DESC"
        
        limit = int(request.args.get('limit', 10))
        query += " LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        weights = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        weights = decimal_to_float(weights)
        
        return jsonify({
            "container_number": container_number,
            "total_records": len(weights),
            "weights": weights
        })
        
    except Exception as e:
        app.logger.error(f"Error getting weight by container: {e}")
        abort(500, description=f"Database error: {str(e)}")


@app.route("/api/weight/truck/<plate_number>", methods=["GET"])
def get_weight_by_truck(plate_number: str):
    """
    Get weight data for a specific truck/plate number
    Query params:
    - limit: int (default 10)
    - start_date: date (optional)
    - end_date: date (optional)
    """
    if connection_pool is None:
        abort(503, description="Database connection not available")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT id, session_id, container_number, truck_id,
                   weight_kg, weight_timestamp, scale_id, operator_name, notes
            FROM weight_logs 
            WHERE plate_number = %s
        """
        params = [plate_number]
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            query += " AND weight_timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND weight_timestamp <= %s"
            params.append(end_date)
            
        query += " ORDER BY weight_timestamp DESC"
        
        limit = int(request.args.get('limit', 10))
        query += " LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        weights = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        weights = decimal_to_float(weights)
        
        return jsonify({
            "plate_number": plate_number,
            "total_records": len(weights),
            "weights": weights
        })
        
    except Exception as e:
        app.logger.error(f"Error getting weight by truck: {e}")
        abort(500, description=f"Database error: {str(e)}")


@app.route("/api/weight/statistics", methods=["GET"])
def get_weight_statistics():
    """
    Get weight statistics for a date range
    Query params:
    - start_date: date (required)
    - end_date: date (required)
    - group_by: str (day/hour/container) - grouping method
    """
    if connection_pool is None:
        abort(503, description="Database connection not available")
    
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date or not end_date:
            abort(400, description="start_date and end_date are required")
        
        group_by = request.args.get('group_by', 'day')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if group_by == 'hour':
            group_clause = "DATE_FORMAT(weight_timestamp, '%Y-%m-%d %H:00:00')"
        elif group_by == 'container':
            group_clause = "container_number"
        else:
            group_clause = "DATE(weight_timestamp)"
        
        query = f"""
            SELECT 
                {group_clause} as period,
                COUNT(*) as total_measurements,
                MIN(weight_kg) as min_weight,
                MAX(weight_kg) as max_weight,
                AVG(weight_kg) as avg_weight,
                COUNT(DISTINCT container_number) as unique_containers,
                COUNT(DISTINCT plate_number) as unique_trucks
            FROM weight_logs 
            WHERE weight_timestamp >= %s AND weight_timestamp <= %s
            GROUP BY period
            ORDER BY period DESC
        """
        
        cursor.execute(query, (start_date, end_date))
        statistics = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        statistics = decimal_to_float(statistics)
        
        return jsonify({
            "start_date": start_date,
            "end_date": end_date,
            "group_by": group_by,
            "statistics": statistics
        })
        
    except Exception as e:
        app.logger.error(f"Error getting weight statistics: {e}")
        abort(500, description=f"Database error: {str(e)}")


@app.route("/api/weight", methods=["POST"])
def create_weight_record():
    """
    Create a new weight record
    Body: JSON with weight data
    Required fields: session_id, weight_kg
    Optional: container_number, truck_id, plate_number, weight_timestamp, scale_id, operator_name, notes
    """
    if connection_pool is None:
        abort(503, description="Database connection not available")
    
    try:
        data = request.get_json()
        
        if not data:
            abort(400, description="Data weight tidak ditemukan")
        
        required_fields = ['session_id', 'weight_kg']
        for field in required_fields:
            if field not in data:
                abort(400, description=f"Field '{field}' is required")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            INSERT INTO weight_logs 
            (session_id, container_number, truck_id, plate_number, 
             weight_kg, weight_timestamp, scale_id, operator_name, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        timestamp = data.get('weight_timestamp')
        if not timestamp:
            timestamp = datetime.now()
        
        values = (
            data['session_id'],
            data.get('container_number'),
            data.get('truck_id'),
            data.get('plate_number'),
            data['weight_kg'],
            timestamp,
            data.get('scale_id'),
            data.get('operator_name'),
            data.get('notes')
        )
        
        cursor.execute(query, values)
        conn.commit()
        
        new_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Weight record created successfully",
            "id": new_id
        }), 201
        
    except Exception as e:
        app.logger.error(f"Error creating weight record: {e}")
        abort(500, description=f"Database error: {str(e)}")


@app.route("/api/sessions/<date>/<session_id>/complete", methods=["GET"])
def get_complete_session_data(date: str, session_id: str):
    """Get complete session data combining OCR detections and weight data"""
    # Get OCR/detection data
    sess_dir = DEBUG_ROOT / date / f"session_{session_id}"
    
    ocr_data = []
    if sess_dir.is_dir():
        for det_dir in sorted(sess_dir.iterdir()):
            if det_dir.is_dir():
                pairs = _list_image_pairs(det_dir)
                for item in pairs:
                    item["detection_type"] = det_dir.name
                    ocr_data.append(item)
        
        ocr_data.sort(key=lambda x: x.get("ts_ms", 0), reverse=True)
    
    # Get weight data from database
    weight_data = {"has_weight_data": False, "records": [], "total_records": 0}
    
    if connection_pool is not None:
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT id, container_number, truck_id, plate_number, 
                       weight_kg, weight_timestamp, scale_id, operator_name, notes
                FROM weight_logs 
                WHERE session_id = %s 
                ORDER BY weight_timestamp DESC
            """
            cursor.execute(query, (session_id,))
            weight_records = cursor.fetchall()
            cursor.close()
            conn.close()
            
            weight_data = {
                "has_weight_data": len(weight_records) > 0,
                "records": decimal_to_float(weight_records),
                "total_records": len(weight_records)
            }
            
            if len(weight_records) > 0:
                weight_values = [float(w['weight_kg']) for w in weight_records if w['weight_kg']]
                if weight_values:
                    weight_data["statistics"] = {
                        "min_weight_kg": min(weight_values),
                        "max_weight_kg": max(weight_values),
                        "avg_weight_kg": sum(weight_values) / len(weight_values)
                    }
                    
        except Exception as e:
            app.logger.error(f"Error getting weight data for complete view: {e}")
            weight_data = {"has_weight_data": False, "error": str(e)}
    
    return jsonify({
        "date": date,
        "session_id": session_id,
        "ocr_detections": ocr_data,
        "total_detections": len(ocr_data),
        "weight_data": weight_data
    })


# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


# ============ MAIN ============

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG OCR API WITH WEIGHT INTEGRATION")
    print("=" * 60)
    print(f"DEBUG_OCR_ROOT : {DEBUG_ROOT.resolve()}")
    print(f"Exists         : {DEBUG_ROOT.is_dir()}")
    print(f"Database       : {'Connected' if connection_pool else 'Not Connected'}")
    print("=" * 60)
    print("Starting Flask API  →  http://0.0.0.0:5050")
    print("\n📁 OCR ENDPOINTS:")
    print("  GET /api/sessions")
    print("  GET /api/sessions/<date>")
    print("  GET /api/sessions/<date>/<session_id>")
    print("  GET /api/sessions/<date>/<session_id>/detections")
    print("  GET /api/sessions/<date>/<session_id>/detections/<detection_type>")
    print("  GET /api/sessions/<date>/<session_id>/detection-types")
    print("  GET /api/sessions/<date>/<session_id>/images/<detection_type>")
    print("  GET /api/sessions/<date>/<session_id>/summary")
    print("  GET /api/sessions/<date>/<session_id>/image/<detection_type>/<filename>")
    print("\n⚖️ WEIGHT ENDPOINTS:")
    print("  GET /api/sessions/<date>/<session_id>/weight")
    print("  GET /api/sessions/<date>/<session_id>/complete")
    print("  GET /api/weight/container/<container_number>")
    print("  GET /api/weight/truck/<plate_number>")
    print("  GET /api/weight/statistics")
    print("  POST /api/weight")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5050, debug=True)