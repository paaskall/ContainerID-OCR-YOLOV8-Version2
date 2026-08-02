import mysql.connector
from mysql.connector import Error, pooling
import os
import sys
from datetime import datetime
import logging

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

try:
    from config.database_config import db_config
except ImportError:
    class FallbackDBConfig:
        MYSQL_HOST = "localhost"
        MYSQL_PORT = 3306
        MYSQL_USER = "autogate_user"
        MYSQL_PASSWORD = "AutoGate123!"
        MYSQL_DATABASE = "autogate_system"
        POOL_SIZE = 5

        VEHICLES_TABLE = "vehicles"
        CONTAINERS_TABLE = "containers"
        TRUCKS_TABLE = "trucks"
        ACCESS_LOGS_TABLE = "access_logs"
        DETECTION_LOGS_TABLE = "detection_logs"
        WEIGHT_LOGS_TABLE = "weight_logs"

        WEIGH_SESSIONS_TABLE = "weigh_sessions"
        WEIGH_SESSION_DETECTIONS_TABLE = "weigh_session_detections"
        CONTAINER_TRIPS_TABLE = "container_trips"

    db_config = FallbackDBConfig()

if not hasattr(db_config, "WEIGH_SESSIONS_TABLE"):
    db_config.WEIGH_SESSIONS_TABLE = "weigh_sessions"
if not hasattr(db_config, "WEIGH_SESSION_DETECTIONS_TABLE"):
    db_config.WEIGH_SESSION_DETECTIONS_TABLE = "weigh_session_detections"
if not hasattr(db_config, "CONTAINER_TRIPS_TABLE"):
    db_config.CONTAINER_TRIPS_TABLE = "container_trips"

try:
    from src.core.iso_6346 import validate_iso as _validate_iso_for_db
except Exception:
    _validate_iso_for_db = None
    print("[DB] Warning: iso_6346 not available, container validation in DB handler will be skipped.")


def _is_iso_valid(container_number: str) -> bool:
    """Helper: cek apakah container_number valid ISO 6346."""
    if _validate_iso_for_db is None:
        return True
    try:
        result = _validate_iso_for_db(container_number)
        return bool(getattr(result, "is_valid", False))
    except Exception:
        return True


class MySQLDatabaseHandler:
    def __init__(self):
        self.connection_pool = None
        self.setup_connection_pool()
        self.create_tables()

    def setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name="autogate_pool",
                pool_size=getattr(db_config, "POOL_SIZE", 5),
                pool_reset_session=True,
                host=db_config.MYSQL_HOST,
                port=getattr(db_config, "MYSQL_PORT", 3306),
                user=db_config.MYSQL_USER,
                password=db_config.MYSQL_PASSWORD,
                database=db_config.MYSQL_DATABASE,
                autocommit=True
            )
            print("MySQL connection pool created successfully")
        except Error as e:
            print(f"Error creating connection pool: {e}")
            self.connection_pool = None

    def get_connection(self):
        """Get connection from pool"""
        if self.connection_pool:
            try:
                return self.connection_pool.get_connection()
            except Error as e:
                print(f"Error getting connection: {e}")
        return None

    def _table_exists(self, cursor, table_name: str) -> bool:
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (db_config.MYSQL_DATABASE, table_name))
        return cursor.fetchone()[0] > 0

    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """, (db_config.MYSQL_DATABASE, table_name, column_name))
        return cursor.fetchone()[0] > 0

    def _try_add_column(self, cursor, table: str, ddl: str):
        try:
            cursor.execute(f"ALTER TABLE {table} {ddl}")
        except Error as e:
            if getattr(e, "errno", None) == 1060:
                return
            print(f"ALTER TABLE failed ({table}): {e}")

    def _try_add_index(self, cursor, table: str, ddl: str):
        try:
            cursor.execute(f"ALTER TABLE {table} {ddl}")
        except Error:
            pass

    def create_tables(self):
        """Create all necessary tables + new weigh session schema"""
        connection = self.get_connection()
        if not connection:
            print("No database connection available")
            return

        cursor = None
        try:
            cursor = connection.cursor()

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.VEHICLES_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_number VARCHAR(20) UNIQUE NOT NULL,
                    vehicle_type VARCHAR(50),
                    max_weight DECIMAL(10,2),
                    owner_name VARCHAR(100),
                    company VARCHAR(100),
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status ENUM('active', 'inactive', 'blocked') DEFAULT 'active',
                    INDEX idx_plate_number (plate_number),
                    INDEX idx_status (status)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.CONTAINERS_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    container_number VARCHAR(20) UNIQUE NOT NULL,
                    container_type VARCHAR(50),
                    size VARCHAR(20),
                    owner_company VARCHAR(100),
                    current_status ENUM('empty', 'loaded', 'in_transit', 'delivered') DEFAULT 'empty',
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_container_number (container_number),
                    INDEX idx_status (current_status)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.TRUCKS_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    truck_id VARCHAR(50) UNIQUE NOT NULL,
                    plate_number VARCHAR(20),
                    driver_name VARCHAR(100),
                    company VARCHAR(100),
                    max_container_capacity INT,
                    current_location VARCHAR(100),
                    status ENUM('active', 'maintenance', 'inactive') DEFAULT 'active',
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (plate_number) REFERENCES {db_config.VEHICLES_TABLE}(plate_number),
                    INDEX idx_truck_id (truck_id),
                    INDEX idx_plate_number (plate_number)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.ACCESS_LOGS_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    camera_id INT NOT NULL,
                    detection_type ENUM('Container_ID', 'Container_ID_Vertikal', 'truck_id', 'plate_number'),
                    detected_text VARCHAR(100),
                    confidence DECIMAL(5,4),
                    access_granted BOOLEAN DEFAULT FALSE,
                    reason TEXT,
                    weight_measured DECIMAL(10,2),
                    container_number VARCHAR(20),
                    truck_id VARCHAR(50),
                    plate_number VARCHAR(20),
                    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_camera_id (camera_id),
                    INDEX idx_detection_type (detection_type),
                    INDEX idx_timestamp (log_timestamp),
                    INDEX idx_container_number (container_number),
                    INDEX idx_truck_id (truck_id),
                    FOREIGN KEY (container_number) REFERENCES {db_config.CONTAINERS_TABLE}(container_number),
                    FOREIGN KEY (truck_id) REFERENCES {db_config.TRUCKS_TABLE}(truck_id),
                    FOREIGN KEY (plate_number) REFERENCES {db_config.VEHICLES_TABLE}(plate_number)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.DETECTION_LOGS_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    camera_id INT NOT NULL,
                    detection_type ENUM('Container_ID', 'Container_ID_Vertikal', 'truck_id', 'plate_number'),
                    detected_text VARCHAR(100),
                    confidence DECIMAL(5,4),
                    roi_coordinates TEXT,
                    frame_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_duration DECIMAL(8,4),
                    ocr_result TEXT,
                    INDEX idx_camera_id (camera_id),
                    INDEX idx_detection_type (detection_type),
                    INDEX idx_timestamp (frame_timestamp),
                    INDEX idx_detected_text (detected_text(20))
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.WEIGHT_LOGS_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    container_number VARCHAR(20),
                    truck_id VARCHAR(50),
                    plate_number VARCHAR(20),
                    weight_kg DECIMAL(10,2),
                    weight_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scale_id VARCHAR(50),
                    operator_name VARCHAR(100),
                    notes TEXT,
                    FOREIGN KEY (container_number) REFERENCES {db_config.CONTAINERS_TABLE}(container_number),
                    FOREIGN KEY (truck_id) REFERENCES {db_config.TRUCKS_TABLE}(truck_id),
                    FOREIGN KEY (plate_number) REFERENCES {db_config.VEHICLES_TABLE}(plate_number),
                    INDEX idx_container_number (container_number),
                    INDEX idx_truck_id (truck_id),
                    INDEX idx_timestamp (weight_timestamp)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.WEIGH_SESSIONS_TABLE} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    camera_id INT,
                    scale_id VARCHAR(50),
                    weight_raw VARCHAR(50),
                    weight_kg DECIMAL(10,2),

                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP NULL,

                    -- Legacy summary merged field
                    best_container VARCHAR(20) NULL,

                    -- Truck/plate
                    best_truck_id VARCHAR(50) NULL,
                    best_plate_number VARCHAR(20) NULL,

                    -- Split container fields
                    container_id_h VARCHAR(20) NULL,
                    container_id_v VARCHAR(20) NULL,

                    -- ISO 6346 summary fields (diisi dari detection level via aggregate)
                    -- NOTE: field ini opsional dan saat ini tidak diisi otomatis oleh session_aggregator.
                    -- Jika ingin diisi, harus ada logic aggregate dari weigh_session_detections.
                    iso_owner_code VARCHAR(4) NULL,
                    iso_category_id CHAR(1) NULL,
                    iso_serial VARCHAR(6) NULL,
                    iso_check_digit CHAR(1) NULL,
                    iso_is_valid BOOLEAN NULL,

                    notes TEXT,

                    INDEX idx_started_at (started_at),
                    INDEX idx_best_container (best_container),
                    INDEX idx_best_truck (best_truck_id),
                    INDEX idx_best_plate (best_plate_number),
                    INDEX idx_container_id_h (container_id_h),
                    INDEX idx_container_id_v (container_id_v),

                    FOREIGN KEY (best_container) REFERENCES {db_config.CONTAINERS_TABLE}(container_number),
                    FOREIGN KEY (best_truck_id) REFERENCES {db_config.TRUCKS_TABLE}(truck_id),
                    FOREIGN KEY (best_plate_number) REFERENCES {db_config.VEHICLES_TABLE}(plate_number)
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.WEIGH_SESSION_DETECTIONS_TABLE} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id BIGINT NOT NULL,

                    camera_id INT NULL,
                    detection_type ENUM('Container_ID', 'Container_ID_Vertikal', 'truck_id', 'plate_number') NOT NULL,
                    ocr_text VARCHAR(100) NULL,
                    confidence DECIMAL(5,4) NULL,

                    x1 INT NULL,
                    y1 INT NULL,
                    x2 INT NULL,
                    y2 INT NULL,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    INDEX idx_session_id (session_id),
                    INDEX idx_detection_type (detection_type),
                    INDEX idx_created_at (created_at),

                    FOREIGN KEY (session_id) REFERENCES {db_config.WEIGH_SESSIONS_TABLE}(id)
                        ON DELETE CASCADE
                )
            ''')

            if self._table_exists(cursor, db_config.WEIGH_SESSION_DETECTIONS_TABLE):
                t = db_config.WEIGH_SESSION_DETECTIONS_TABLE

                def add_col_if_missing(col, ddl):
                    if not self._column_exists(cursor, t, col):
                        self._try_add_column(cursor, t, ddl)

                add_col_if_missing("iso_is_valid",       "ADD COLUMN iso_is_valid BOOLEAN NULL")
                add_col_if_missing("iso_owner_code",     "ADD COLUMN iso_owner_code VARCHAR(4) NULL")
                add_col_if_missing("iso_category_id",    "ADD COLUMN iso_category_id CHAR(1) NULL")
                add_col_if_missing("iso_serial",         "ADD COLUMN iso_serial VARCHAR(6) NULL")
                add_col_if_missing("iso_check_digit",    "ADD COLUMN iso_check_digit CHAR(1) NULL")
                add_col_if_missing("iso_calc_digit",     "ADD COLUMN iso_calc_digit CHAR(1) NULL")
                add_col_if_missing("iso_reason",         "ADD COLUMN iso_reason VARCHAR(80) NULL")
                add_col_if_missing("iso_repaired_text",  "ADD COLUMN iso_repaired_text VARCHAR(20) NULL")
                add_col_if_missing("iso_repair_score",   "ADD COLUMN iso_repair_score DECIMAL(6,4) NULL")
                add_col_if_missing("iso_repair_edits",   "ADD COLUMN iso_repair_edits INT NULL")

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {db_config.CONTAINER_TRIPS_TABLE} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id BIGINT NOT NULL,

                    container_number VARCHAR(20) NOT NULL,

                    container_id_h VARCHAR(20) NULL,
                    container_id_v VARCHAR(20) NULL,

                    truck_id VARCHAR(50) NULL,
                    plate_number VARCHAR(20) NULL,

                    weight_kg DECIMAL(10,2) NULL,
                    scale_id VARCHAR(50) NULL,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    INDEX idx_container_number (container_number),
                    INDEX idx_container_id_h (container_id_h),
                    INDEX idx_container_id_v (container_id_v),
                    INDEX idx_truck_id (truck_id),
                    INDEX idx_plate_number (plate_number),
                    INDEX idx_session_id (session_id),

                    FOREIGN KEY (session_id) REFERENCES {db_config.WEIGH_SESSIONS_TABLE}(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (container_number) REFERENCES {db_config.CONTAINERS_TABLE}(container_number),
                    FOREIGN KEY (truck_id) REFERENCES {db_config.TRUCKS_TABLE}(truck_id),
                    FOREIGN KEY (plate_number) REFERENCES {db_config.VEHICLES_TABLE}(plate_number)
                )
            ''')

            if self._table_exists(cursor, db_config.WEIGHT_LOGS_TABLE):
                if not self._column_exists(cursor, db_config.WEIGHT_LOGS_TABLE, "session_id"):
                    self._try_add_column(cursor, db_config.WEIGHT_LOGS_TABLE, "ADD COLUMN session_id BIGINT NULL")
                    self._try_add_index(cursor, db_config.WEIGHT_LOGS_TABLE, "ADD INDEX idx_session_id (session_id)")
                    try:
                        cursor.execute(f"""
                            ALTER TABLE {db_config.WEIGHT_LOGS_TABLE}
                            ADD CONSTRAINT fk_weight_logs_session
                            FOREIGN KEY (session_id) REFERENCES {db_config.WEIGH_SESSIONS_TABLE}(id)
                            ON DELETE SET NULL
                        """)
                    except Error:
                        pass

            if self._table_exists(cursor, db_config.WEIGH_SESSIONS_TABLE):
                if not self._column_exists(cursor, db_config.WEIGH_SESSIONS_TABLE, "container_id_h"):
                    self._try_add_column(cursor, db_config.WEIGH_SESSIONS_TABLE, "ADD COLUMN container_id_h VARCHAR(20) NULL")
                    self._try_add_index(cursor, db_config.WEIGH_SESSIONS_TABLE, "ADD INDEX idx_container_id_h (container_id_h)")
                if not self._column_exists(cursor, db_config.WEIGH_SESSIONS_TABLE, "container_id_v"):
                    self._try_add_column(cursor, db_config.WEIGH_SESSIONS_TABLE, "ADD COLUMN container_id_v VARCHAR(20) NULL")
                    self._try_add_index(cursor, db_config.WEIGH_SESSIONS_TABLE, "ADD INDEX idx_container_id_v (container_id_v)")

            if self._table_exists(cursor, db_config.CONTAINER_TRIPS_TABLE):
                if not self._column_exists(cursor, db_config.CONTAINER_TRIPS_TABLE, "container_id_h"):
                    self._try_add_column(cursor, db_config.CONTAINER_TRIPS_TABLE, "ADD COLUMN container_id_h VARCHAR(20) NULL")
                    self._try_add_index(cursor, db_config.CONTAINER_TRIPS_TABLE, "ADD INDEX idx_container_id_h (container_id_h)")
                if not self._column_exists(cursor, db_config.CONTAINER_TRIPS_TABLE, "container_id_v"):
                    self._try_add_column(cursor, db_config.CONTAINER_TRIPS_TABLE, "ADD COLUMN container_id_v VARCHAR(20) NULL")
                    self._try_add_index(cursor, db_config.CONTAINER_TRIPS_TABLE, "ADD INDEX idx_container_id_v (container_id_v)")

            print("All database tables created/verified (including weigh_sessions split container fields)")

        except Error as e:
            print(f"Error creating tables: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if connection and connection.is_connected():
                    if cursor:
                        cursor.close()
                    connection.close()
            except Exception:
                pass

    def ensure_vehicle_exists(self, plate_number: str):
        if not plate_number:
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT plate_number FROM {db_config.VEHICLES_TABLE} WHERE plate_number=%s",
                (plate_number,)
            )
            if cur.fetchone() is None:
                cur.execute(
                    f"INSERT INTO {db_config.VEHICLES_TABLE}(plate_number, vehicle_type, status) VALUES(%s,'Truck','active')",
                    (plate_number,)
                )
        except Error:
            pass
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def ensure_truck_exists(self, truck_id: str, plate_number: str = None):
        if not truck_id:
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT truck_id FROM {db_config.TRUCKS_TABLE} WHERE truck_id=%s",
                (truck_id,)
            )
            if cur.fetchone() is None:
                if plate_number:
                    self.ensure_vehicle_exists(plate_number)
                cur.execute(
                    f"INSERT INTO {db_config.TRUCKS_TABLE}(truck_id, plate_number, status) VALUES(%s,%s,'active')",
                    (truck_id, plate_number)
                )
        except Error:
            pass
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def ensure_container_exists(self, container_number: str):
        """Buat record master container jika belum ada."""
        if not container_number:
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT container_number FROM {db_config.CONTAINERS_TABLE} WHERE container_number=%s",
                (container_number,)
            )
            if cur.fetchone() is None:
                cur.execute(
                    f"INSERT INTO {db_config.CONTAINERS_TABLE}(container_number, container_type, size) VALUES(%s,'Dry','40ft')",
                    (container_number,)
                )
        except Error:
            pass
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def create_weigh_session(
        self,
        camera_id: int,
        scale_id: str,
        weight_raw: str,
        weight_kg=None,
        notes: str = None,
        session_id: int = None,
    ):
        conn = self.get_connection()
        if not conn:
            print("[DB] create_weigh_session: no connection")
            return None

        try:
            cur = conn.cursor()

            if session_id is None:
                cur.execute(f"""
                    INSERT INTO {db_config.WEIGH_SESSIONS_TABLE}
                    (camera_id, scale_id, weight_raw, weight_kg, notes)
                    VALUES (%s, %s, %s, %s, %s)
                """, (camera_id, scale_id, weight_raw, weight_kg, notes))
                sid = cur.lastrowid
            else:
                cur.execute(f"""
                    INSERT INTO {db_config.WEIGH_SESSIONS_TABLE}
                    (id, camera_id, scale_id, weight_raw, weight_kg, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (session_id, camera_id, scale_id, weight_raw, weight_kg, notes))
                sid = session_id

            return sid

        except Error as e:
            print(f"[DB] create_weigh_session ERROR | errno={getattr(e,'errno',None)} | {e}")
            return None
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def add_session_detection(
        self,
        session_id: int,
        camera_id: int,
        detection_type: str,
        ocr_text: str,
        confidence: float,
        bbox: tuple,
    ):
        if not session_id:
            print("[DB] add_session_detection: invalid session_id")
            return None

        x1, y1, x2, y2 = bbox if bbox and len(bbox) == 4 else (None, None, None, None)

        conn = self.get_connection()
        if not conn:
            print("[DB] add_session_detection: no connection")
            return None

        try:
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {db_config.WEIGH_SESSION_DETECTIONS_TABLE}
                (session_id, camera_id, detection_type, ocr_text, confidence, x1, y1, x2, y2)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, camera_id, detection_type, ocr_text, confidence, x1, y1, x2, y2))
            return cur.lastrowid
        except Error as e:
            print(f"[DB] add_session_detection ERROR | errno={getattr(e,'errno',None)} | {e}")
            return None
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def update_detection_iso(
        self,
        detection_id: int,
        iso_is_valid=None,
        iso_owner_code: str = None,
        iso_category_id: str = None,
        iso_serial: str = None,
        iso_check_digit: str = None,
        iso_calc_digit: str = None,
        iso_reason: str = None,
        iso_repaired_text: str = None,
        iso_repair_score=None,
        iso_repair_edits: int = None,
    ):
        if not detection_id:
            return False

        conn = self.get_connection()
        if not conn:
            return False

        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                UPDATE {db_config.WEIGH_SESSION_DETECTIONS_TABLE}
                SET
                    iso_is_valid     = %s,
                    iso_owner_code   = %s,
                    iso_category_id  = %s,
                    iso_serial       = %s,
                    iso_check_digit  = %s,
                    iso_calc_digit   = %s,
                    iso_reason       = %s,
                    iso_repaired_text = %s,
                    iso_repair_score = %s,
                    iso_repair_edits = %s
                WHERE id = %s
                """,
                (
                    iso_is_valid, iso_owner_code, iso_category_id, iso_serial,
                    iso_check_digit, iso_calc_digit, iso_reason,
                    iso_repaired_text, iso_repair_score, iso_repair_edits,
                    detection_id,
                ),
            )
            return True
        except Error as e:
            print(f"[DB] update_detection_iso ERROR | errno={getattr(e,'errno',None)} | {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def update_weigh_session_container_fields(
        self,
        session_id: int,
        container_id_h: str = None,
        container_id_v: str = None,
        notes_append: str = None,
    ):
        if not session_id:
            return False

        conn = self.get_connection()
        if not conn:
            return False

        try:
            cur = conn.cursor()

            if notes_append:
                cur.execute(
                    f"SELECT notes FROM {db_config.WEIGH_SESSIONS_TABLE} WHERE id=%s",
                    (session_id,)
                )
                row = cur.fetchone()
                existing = row[0] if row and row[0] else ""
                merged_notes = (existing + "\n" + notes_append).strip() if existing else notes_append
            else:
                merged_notes = None

            cur.execute(
                f"""
                UPDATE {db_config.WEIGH_SESSIONS_TABLE}
                SET
                    container_id_h = COALESCE(%s, container_id_h),
                    container_id_v = COALESCE(%s, container_id_v),
                    notes = COALESCE(%s, notes)
                WHERE id = %s
                """,
                (container_id_h, container_id_v, merged_notes, session_id)
            )
            return True

        except Error as e:
            print(f"Error update_weigh_session_container_fields: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def update_weigh_session_summary(
        self,
        session_id: int,
        best_container: str = None,
        best_truck_id: str = None,
        best_plate_number: str = None,
        iso_owner_code: str = None,
        iso_category_id: str = None,
        iso_serial: str = None,
        iso_check_digit: str = None,
        iso_is_valid=None,
        notes_append: str = None,
    ):
        """Update best results + optional ISO6346 summary fields."""
        if not session_id:
            return False

        if best_container:
            self.ensure_container_exists(best_container)
        if best_plate_number:
            self.ensure_vehicle_exists(best_plate_number)
        if best_truck_id:
            self.ensure_truck_exists(best_truck_id, plate_number=best_plate_number)

        conn = self.get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()

            if notes_append:
                cur.execute(
                    f"SELECT notes FROM {db_config.WEIGH_SESSIONS_TABLE} WHERE id=%s",
                    (session_id,)
                )
                row = cur.fetchone()
                existing = row[0] if row and row[0] else ""
                merged = (existing + "\n" + notes_append).strip() if existing else notes_append
            else:
                merged = None

            cur.execute(
                f"""
                UPDATE {db_config.WEIGH_SESSIONS_TABLE}
                SET
                    best_container    = COALESCE(%s, best_container),
                    best_truck_id     = COALESCE(%s, best_truck_id),
                    best_plate_number = COALESCE(%s, best_plate_number),

                    iso_owner_code    = COALESCE(%s, iso_owner_code),
                    iso_category_id   = COALESCE(%s, iso_category_id),
                    iso_serial        = COALESCE(%s, iso_serial),
                    iso_check_digit   = COALESCE(%s, iso_check_digit),
                    iso_is_valid      = COALESCE(%s, iso_is_valid),

                    notes = COALESCE(%s, notes)
                WHERE id = %s
                """,
                (
                    best_container, best_truck_id, best_plate_number,
                    iso_owner_code, iso_category_id, iso_serial, iso_check_digit, iso_is_valid,
                    merged,
                    session_id,
                )
            )
            return True
        except Error as e:
            print(f"Error update_weigh_session_summary: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def close_weigh_session(self, session_id: int):
        if not session_id:
            return False
        conn = self.get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE {db_config.WEIGH_SESSIONS_TABLE} SET ended_at = CURRENT_TIMESTAMP WHERE id=%s",
                (session_id,)
            )
            return True
        except Error as e:
            print(f"Error close_weigh_session: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def link_container_trip(
        self,
        session_id: int,
        container_number: str,
        container_id_h: str = None,
        container_id_v: str = None,
        truck_id: str = None,
        plate_number: str = None,
        weight_kg=None,
        scale_id: str = None,
        skip_iso_check: bool = False,
    ):
        """Catat trip container per session."""
        if not session_id or not container_number:
            return False

        if not skip_iso_check:
            if not _is_iso_valid(container_number):
                print(
                    f"[DB] link_container_trip SKIPPED: container_number '{container_number}' "
                    f"tidak valid ISO 6346. Record master container TIDAK dibuat. "
                    f"Raw OCR tetap tersimpan di weigh_session_detections."
                )
                return False

        self.ensure_container_exists(container_number)
        if plate_number:
            self.ensure_vehicle_exists(plate_number)
        if truck_id:
            self.ensure_truck_exists(truck_id, plate_number=plate_number)

        conn = self.get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {db_config.CONTAINER_TRIPS_TABLE}
                (session_id, container_number, container_id_h, container_id_v,
                 truck_id, plate_number, weight_kg, scale_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (session_id, container_number, container_id_h, container_id_v,
                 truck_id, plate_number, weight_kg, scale_id)
            )
            return True
        except Error as e:
            print(f"Error link_container_trip: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def log_weight(
        self,
        container_number=None,
        truck_id=None,
        plate_number=None,
        weight_kg=0,
        scale_id="",
        operator_name="",
        notes="",
        session_id=None,
    ):
        """Log weight measurement."""
        conn = self.get_connection()
        if not conn:
            return False

        if container_number:
            self.ensure_container_exists(container_number)
        if plate_number:
            self.ensure_vehicle_exists(plate_number)
        if truck_id:
            self.ensure_truck_exists(truck_id, plate_number=plate_number)

        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    f'''
                    INSERT INTO {db_config.WEIGHT_LOGS_TABLE}
                    (container_number, truck_id, plate_number, weight_kg,
                     scale_id, operator_name, notes, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ''',
                    (container_number, truck_id, plate_number, weight_kg,
                     scale_id, operator_name, notes, session_id)
                )
            except Error:
                cur.execute(
                    f'''
                    INSERT INTO {db_config.WEIGHT_LOGS_TABLE}
                    (container_number, truck_id, plate_number, weight_kg,
                     scale_id, operator_name, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''',
                    (container_number, truck_id, plate_number, weight_kg,
                     scale_id, operator_name, notes)
                )
            return True
        except Error as e:
            print(f"Error logging weight: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def log_detection(
        self, camera_id, detection_type, detected_text, confidence,
        roi_coordinates=None, processed_duration=0, ocr_result=""
    ):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            roi_str = ""
            if roi_coordinates:
                roi_str = roi_coordinates if isinstance(roi_coordinates, str) else str(roi_coordinates)

            cur.execute(f'''
                INSERT INTO {db_config.DETECTION_LOGS_TABLE}
                (camera_id, detection_type, detected_text, confidence,
                 roi_coordinates, processed_duration, ocr_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (camera_id, detection_type, detected_text, confidence,
                  roi_str, processed_duration, ocr_result))
            return True
        except Error as e:
            print(f"Error logging detection: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    def log_access(
        self, camera_id, detection_type, detected_text, confidence,
        access_granted, reason="", weight_measured=None,
        container_number=None, truck_id=None, plate_number=None,
    ):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute(f'''
                INSERT INTO {db_config.ACCESS_LOGS_TABLE}
                (camera_id, detection_type, detected_text, confidence, access_granted, reason,
                 weight_measured, container_number, truck_id, plate_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (camera_id, detection_type, detected_text, confidence, access_granted, reason,
                  weight_measured, container_number, truck_id, plate_number))
            return True
        except Error as e:
            print(f"Error logging access: {e}")
            return False
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass


db_handler = MySQLDatabaseHandler()