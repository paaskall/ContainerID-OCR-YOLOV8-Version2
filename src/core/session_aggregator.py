import time
import threading
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

INVALID_HARD = {"ERROR", "ROI_INVALID", "", None}

import re as _re
_NOISE_PATTERN = _re.compile(
    r"GATE|TPKM|GATEIN|GATEOUT|GATE\s*IN|GATE\s*OUT"
    r"|^(MON|TUE|WED|THU|FRI|SAT|SUN)"
    r"|^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
    _re.IGNORECASE,
)

_CONTAINER_LOOSE_RE = _re.compile(r"^[A-Z]{3}")

MAX_OBS = 5000
_MAX_SEEN_KEYS = 10_000

try:
    from src.core.iso_6346 import validate_iso, repair_iso
except Exception as e:
    validate_iso = None
    repair_iso = None
    print(f"[SESSION] iso_6346 import failed: {e}")


@dataclass
class OcrObservation:
    detection_type: str
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    ts: float
    camera_id: Optional[int] = None

    iso_valid: bool = False
    iso_repaired: Optional[str] = None
    iso_score: float = 0.0
    iso_reason: Optional[str] = None


def _extract_owner_code(text: str) -> Optional[str]:
    """Ambil 3-4 huruf pertama container ID sebagai owner code penanda truk."""
    m = _re.match(r'^([A-Z]{3,4})', (text or "").strip().upper())
    return m.group(1) if m else None


def _owner_codes_match(a: Optional[str], b: Optional[str]) -> bool:
    """Treat 3-letter vs 4-letter variants with the same prefix as the same owner."""
    if not a or not b:
        return False
    a = a.strip().upper()
    b = b.strip().upper()
    if a == b:
        return True
    return len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]


class SessionAggregator:

    def __init__(
        self,
        db_handler=None,
        session_window_sec: float = 30.0,           # 20→30
        max_keepalive_sec: float = 90.0,            # 120→90
        min_gap_between_sessions_sec: float = 1.0,  # 3→1
    ):
        self.db     = db_handler
        self.window = float(session_window_sec)
        self.max_keepalive_sec             = float(max_keepalive_sec)
        self.min_gap_between_sessions_sec  = float(min_gap_between_sessions_sec)

        self.active_session_id: Optional[int] = None
        self.active_until:  float = 0.0
        self._session_started_at: float = 0.0
        self._last_ended_at: float = 0.0

        self.scale_id:    str             = "jembatan_timbangan_2"
        self.weight_raw:  Optional[str]   = None
        self.weight_kg:   Optional[float] = None
        self.camera_id:   Optional[int]   = None

        self._obs:        List[OcrObservation] = []
        self._seen_keys:  set                  = set()

        self._session_owner_code:  Optional[str] = None
        self._owner_code_votes:    Counter        = Counter()
        self._identity_locked:     bool           = False
        self._IDENTITY_LOCK_MIN_VOTES: int        = 3
        self._IDENTITY_CHANGE_MIN_CONF: float     = 0.65

        # Identity change hanya dicatat sebagai indikasi konflik OCR/operasional.
        # Session tetap ditutup oleh trigger deterministik: scale END, vehicle absent, atau timeout.

        self._lock = threading.Lock()

    def _reset_session_buffers_locked(self):
        self._obs.clear()
        self._seen_keys.clear()
        self._session_owner_code = None
        self._owner_code_votes   = Counter()
        self._identity_locked    = False

    def _create_session_locked(
        self,
        camera_id: int,
        weight_raw: str,
        weight_kg: Optional[float],
        scale_id: str,
        now: Optional[float] = None,
    ) -> Optional[int]:
        now = time.time() if now is None else now

        self.camera_id  = camera_id
        self.weight_raw = weight_raw
        self.weight_kg  = weight_kg
        self.scale_id   = scale_id

        self._session_started_at = now
        self.active_until        = now + self.window

        self._reset_session_buffers_locked()

        sid = None
        if self.db and hasattr(self.db, "create_weigh_session"):
            try:
                sid = self.db.create_weigh_session(
                    camera_id=camera_id,
                    scale_id=scale_id,
                    weight_raw=weight_raw,
                    weight_kg=weight_kg,
                )
            except Exception as e:
                print(f"[SESSION] create_weigh_session failed: {e}")
                sid = None

        if sid is None:
            print("[SESSION] FAILED TO CREATE SESSION IN DB")
            self.active_session_id = None
            self.active_until = 0.0
            self._session_started_at = 0.0
            return None

        self.active_session_id = sid
        print(f"[SESSION] START id={sid} weight={weight_raw} window={self.window}s max={self.max_keepalive_sec}s")
        return sid

    def start(
        self,
        camera_id: int,
        weight_raw: str,
        weight_kg: Optional[float],
        scale_id: str = "jembatan_timbangan_2",
    ) -> Optional[int]:
        with self._lock:
            now = time.time()

            gap = now - self._last_ended_at
            if gap < self.min_gap_between_sessions_sec:
                print(
                    f"[SESSION] start() IGNORED – terlalu cepat setelah session terakhir "
                    f"(gap={gap:.1f}s < min={self.min_gap_between_sessions_sec}s)"
                )
                return None

            if self.active_session_id is not None:
                print(
                    f"[SESSION] start() – ada session aktif id={self.active_session_id}, "
                    f"finalize dulu sebelum buka session baru."
                )
                self._finalize_locked(force=True)

            return self._create_session_locked(
                camera_id=camera_id,
                weight_raw=weight_raw,
                weight_kg=weight_kg,
                scale_id=scale_id,
                now=now,
            )

    def is_active(self) -> bool:
        return self.active_session_id is not None and time.time() < self.active_until

    def keep_alive(self, extra_sec: float = 5.0):
        """Perpanjang session, TAPI tidak boleh melebihi max_keepalive_sec dari"""
        with self._lock:
            if self.active_session_id is None:
                return

            now    = time.time()
            hard_limit = self._session_started_at + self.max_keepalive_sec

            if now >= hard_limit:
                print(
                    f"[SESSION] keep_alive: session id={self.active_session_id} "
                    f"melewati max_keepalive_sec={self.max_keepalive_sec}s, paksa finalize."
                )
                self._finalize_locked(force=True)
                return

            new_until = min(now + float(extra_sec), hard_limit)
            self.active_until = max(self.active_until, new_until)

    def _is_container_type(self, detection_type: str) -> bool:
        return "container" in (detection_type or "").lower()

    def _dedup_key(self, detection_type: str, bbox: Tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = bbox
        if self._is_container_type(detection_type):
            q = 5
            return f"{detection_type}:{x1//q}:{y1//q}:{x2//q}:{y2//q}"
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return f"{detection_type}:{cx//50}:{cy//50}"

    def _try_iso_update(
        self, detection_id: int, text_norm: str
    ) -> Tuple[bool, Optional[str], float, Optional[str]]:
        _default = (False, None, 0.0, None)
        if not detection_id or not self.db:
            return _default
        if validate_iso is None or not hasattr(self.db, "update_detection_iso"):
            return _default

        iso = None
        try:
            iso = validate_iso(text_norm)
        except Exception as e:
            print(f"[SESSION] validate_iso failed: {e}")
            return _default

        rep_text = rep_score = rep_edits = rep_reason = None

        if iso is not None and (not getattr(iso, "is_valid", False)) and repair_iso is not None:
            try:
                rep = repair_iso(text_norm, max_edits=4)
                if rep and getattr(rep, "repaired_text", None):
                    rep_text   = rep.repaired_text
                    rep_score  = float(getattr(rep, "score",  0.0) or 0.0)
                    rep_edits  = int(getattr(rep,   "edits",  0)   or 0)
                    rep_reason = getattr(rep, "reason", None)
            except Exception as e:
                print(f"[SESSION] repair_iso failed: {e}")

        reason = getattr(iso, "reason", None)
        if rep_reason:
            reason = f"{reason}|repair={rep_reason}" if reason else f"repair={rep_reason}"

        try:
            self.db.update_detection_iso(
                detection_id=detection_id,
                iso_is_valid=bool(getattr(iso, "is_valid", False)),
                iso_owner_code=getattr(iso, "owner_code", None),
                iso_category_id=getattr(iso, "category_id", None),
                iso_serial=getattr(iso, "serial", None),
                iso_check_digit=getattr(iso, "check_digit", None),
                iso_calc_digit=getattr(iso, "calc_digit", None),
                iso_reason=reason,
                iso_repaired_text=rep_text,
                iso_repair_score=rep_score,
                iso_repair_edits=rep_edits,
            )
        except Exception as e:
            print(f"[SESSION] update_detection_iso failed: {e}")

        is_valid       = bool(getattr(iso, "is_valid", False))
        effective_score = 1.0 if is_valid else float(rep_score or 0.0)
        return (is_valid, rep_text, effective_score, reason)

    def add_observation(
        self,
        detection_type: str,
        text: str,
        confidence: float,
        bbox,
        camera_id: Optional[int] = None,
    ):
        with self._lock:
            if not self._is_active_locked():
                return
            if text is None:
                return
            if not bbox or len(bbox) != 4:
                return

            text_norm    = str(text).strip().upper().replace(" ", "")
            is_container = self._is_container_type(detection_type)

            if text_norm in INVALID_HARD:
                return
            if (not is_container) and text_norm == "TIDAK_TERBACA":
                return

            if is_container:
                if _NOISE_PATTERN.search(text_norm):
                    print(f"[SESSION] NOISE FILTER: '{text_norm}' | type={detection_type}")
                    return
                if not _CONTAINER_LOOSE_RE.match(text_norm):
                    print(f"[SESSION] FORMAT FILTER: '{text_norm}' | type={detection_type}")
                    return

            key = self._dedup_key(detection_type, bbox)
            if key in self._seen_keys:
                return

            if is_container and float(confidence) >= self._IDENTITY_CHANGE_MIN_CONF:
                owner = _extract_owner_code(text_norm)
                if owner:
                    if not self._identity_locked:
                        self._owner_code_votes[owner] += 1
                        top_owner, top_votes = self._owner_code_votes.most_common(1)[0]
                        if top_votes >= self._IDENTITY_LOCK_MIN_VOTES:
                            self._session_owner_code = top_owner
                            self._identity_locked    = True
                            print(
                                f"[SESSION] Identity LOCKED: '{top_owner}' "
                                f"(votes={top_votes}) | session={self.active_session_id}"
                            )
                    else:
                        if not _owner_codes_match(owner, self._session_owner_code):
                            print(
                                f"[SESSION] Identity conflict observed: "
                                f"'{self._session_owner_code}' → '{owner}' "
                                f"| text={text_norm} conf={confidence:.2f} "
                                f"| session={self.active_session_id} | no auto-finalize"
                            )

            if len(self._seen_keys) < _MAX_SEEN_KEYS:
                self._seen_keys.add(key)

            if len(self._obs) >= MAX_OBS:
                print(f"[SESSION] MAX_OBS ({MAX_OBS}) reached, skipping.")
                return

            obs = OcrObservation(
                detection_type=detection_type,
                text=text_norm,
                confidence=float(confidence) if confidence else 0.0,
                bbox=tuple(bbox),
                ts=time.time(),
                camera_id=camera_id if camera_id is not None else self.camera_id,
            )
            self._obs.append(obs)

            det_id = None
            if self.active_session_id and self.db and hasattr(self.db, "add_session_detection"):
                try:
                    det_id = self.db.add_session_detection(
                        session_id=self.active_session_id,
                        camera_id=(camera_id if camera_id is not None else self.camera_id),
                        detection_type=detection_type,
                        ocr_text=text_norm,
                        confidence=float(confidence) if confidence else 0.0,
                        bbox=bbox,
                    )
                except Exception as e:
                    print(f"[SESSION] add_session_detection failed: {e}")
                    det_id = None

            _det_id_for_iso = det_id if (det_id and is_container) else None

        if _det_id_for_iso:
            iso_valid, iso_repaired, iso_score, iso_reason = self._try_iso_update(
                _det_id_for_iso, text_norm
            )
            obs.iso_valid    = iso_valid
            obs.iso_repaired = iso_repaired
            obs.iso_score    = iso_score
            obs.iso_reason   = iso_reason

    def finalize(self, force: bool = False) -> Optional[Tuple[int, Dict]]:
        with self._lock:
            return self._finalize_locked(force=force)

    def finalize_if_needed(self) -> Optional[Tuple[int, Dict]]:
        return self.finalize(force=False)

    def _finalize_locked(self, force: bool = False) -> Optional[Tuple[int, Dict]]:
        """Versi internal finalize – harus dipanggil saat self._lock sudah dipegang."""
        if self.active_session_id is None:
            return None
        if (not force) and (time.time() < self.active_until):
            return None

        sid     = self.active_session_id
        summary = self._build_summary()

        if self.db and hasattr(self.db, "update_weigh_session_container_fields"):
            try:
                self.db.update_weigh_session_container_fields(
                    session_id=sid,
                    container_id_h=summary.get("best_container_h"),
                    container_id_v=summary.get("best_container_v"),
                    notes_append=f"Finalize session. obs={len(self._obs)}",
                )
            except Exception as e:
                print(f"[SESSION] update_weigh_session_container_fields failed: {e}")

        if self.db and hasattr(self.db, "update_weigh_session_summary"):
            try:
                self.db.update_weigh_session_summary(
                    session_id=sid,
                    best_container=None,
                    best_truck_id=summary.get("best_truck_id"),
                    best_plate_number=summary.get("best_plate_number"),
                    notes_append=None,
                )
            except Exception as e:
                print(f"[SESSION] update_weigh_session_summary failed: {e}")

        container_id_h   = summary.get("best_container_h")
        container_id_v   = summary.get("best_container_v")
        parent_container = container_id_h or container_id_v

        if parent_container and self.db and hasattr(self.db, "link_container_trip"):
            try:
                self.db.link_container_trip(
                    session_id=sid,
                    container_number=parent_container,
                    container_id_h=container_id_h,
                    container_id_v=container_id_v,
                    truck_id=summary.get("best_truck_id"),
                    plate_number=summary.get("best_plate_number"),
                    weight_kg=float(self.weight_kg) if self.weight_kg is not None else None,
                    scale_id=self.scale_id,
                )
            except Exception as e:
                print(f"[SESSION] link_container_trip failed: {e}")

        if self.db and hasattr(self.db, "log_weight"):
            try:
                self.db.log_weight(
                    container_number=parent_container,
                    truck_id=summary.get("best_truck_id"),
                    plate_number=summary.get("best_plate_number"),
                    weight_kg=float(self.weight_kg) if self.weight_kg is not None else 0.0,
                    scale_id=self.scale_id,
                    operator_name="AutoGate System",
                    notes=(
                        f"Session {sid} weight_raw={self.weight_raw} "
                        f"| H={container_id_h} | V={container_id_v}"
                    ),
                    session_id=sid,
                )
            except Exception as e:
                print(f"[SESSION] log_weight failed: {e}")

        if self.db and hasattr(self.db, "close_weigh_session"):
            try:
                self.db.close_weigh_session(sid)
            except Exception as e:
                print(f"[SESSION] close_weigh_session failed: {e}")

        print(f"[SESSION] FINALIZE id={sid} summary={summary}")

        self._last_ended_at      = time.time()
        self.active_session_id   = None
        self.active_until        = 0.0
        self._session_started_at = 0.0
        self._reset_session_buffers_locked()

        return sid, summary

    def _is_active_locked(self) -> bool:
        return self.active_session_id is not None and time.time() < self.active_until

    def _best_by_frequency(self, texts: List[str]) -> Optional[str]:
        cleaned = [t for t in texts if t and (t not in INVALID_HARD) and t != "TIDAK_TERBACA"]
        if not cleaned:
            return None
        return Counter(cleaned).most_common(1)[0][0]

    def _best_iso_aware(self, obs_list: List[OcrObservation]) -> Optional[str]:
        if not obs_list:
            return None

        valid_obs = [o for o in obs_list if o.iso_valid and o.text != "TIDAK_TERBACA"]
        if valid_obs:
            return Counter([o.text for o in valid_obs]).most_common(1)[0][0]

        scored = [o for o in obs_list if (o.iso_score or 0.0) > 0.0 and o.text != "TIDAK_TERBACA"]
        if scored:
            scored.sort(key=lambda o: (o.iso_score, o.confidence), reverse=True)
            return scored[0].text

        return self._best_by_frequency([o.text for o in obs_list])

    def _build_summary(self) -> Dict[str, Optional[str]]:
        by_type_texts: Dict[str, List[str]]              = defaultdict(list)
        by_type_obs:   Dict[str, List[OcrObservation]]   = defaultdict(list)

        for o in self._obs:
            by_type_texts[o.detection_type].append(o.text)
            by_type_obs[o.detection_type].append(o)

        return {
            "best_container_h":  self._best_iso_aware(by_type_obs.get("Container_ID", [])),
            "best_container_v":  self._best_iso_aware(by_type_obs.get("Container_ID_Vertikal", [])),
            "best_truck_id":     self._best_by_frequency(by_type_texts.get("truck_id", [])),
            "best_plate_number": self._best_by_frequency(by_type_texts.get("plate_number", [])),
        }

class VehiclePresenceMonitor:
    def __init__(self, monitored_cameras=None, absent_confirm_sec=3.0, present_confirm_sec=0.5, min_present_sec=2.0):
        import threading
        self.monitored_cameras = monitored_cameras or {3: "primary", 0: "secondary"}
        self.absent_confirm_sec = float(absent_confirm_sec)
        self.present_confirm_sec = float(present_confirm_sec)
        self.min_present_sec = float(min_present_sec)
        self._cam_state = {cam_id: {"state": "unknown", "first_seen": 0.0, "present_since": 0.0, "absent_since": 0.0} for cam_id in self.monitored_cameras}
        self.on_vehicle_absent = None
        self.on_vehicle_present = None
        self._absent_triggered = False
        self._lock = threading.Lock()

    def update(self, camera_id, has_detection):
        import time
        if camera_id not in self._cam_state:
            return
        now = time.time()
        with self._lock:
            st = self._cam_state[camera_id]
            if has_detection:
                if st["first_seen"] == 0.0:
                    st["first_seen"] = now
                st["absent_since"] = 0.0
                if st["state"] != "present":
                    if (now - st["first_seen"]) >= self.present_confirm_sec:
                        prev = st["state"]
                        st["state"] = "present"
                        st["present_since"] = now
                        role = self.monitored_cameras.get(camera_id, "?")
                        print(f"[PRESENCE] cam{camera_id}({role}) → PRESENT")
                        if prev == "absent":
                            self._absent_triggered = False
                            if self.on_vehicle_present:
                                self.on_vehicle_present(camera_id)
            else:
                st["first_seen"] = 0.0
                if st["state"] == "present":
                    if st["absent_since"] == 0.0:
                        st["absent_since"] = now
                    held = now - st["present_since"]
                    gone = now - st["absent_since"]
                    if held >= self.min_present_sec and gone >= self.absent_confirm_sec:
                        st["state"] = "absent"
                        role = self.monitored_cameras.get(camera_id, "?")
                        print(f"[PRESENCE] cam{camera_id}({role}) → ABSENT (held={held:.1f}s, gone={gone:.1f}s)")
                        if not self._absent_triggered:
                            self._absent_triggered = True
                            if self.on_vehicle_absent:
                                self.on_vehicle_absent(camera_id, role)
                elif st["state"] == "unknown":
                    st["state"] = "absent"

    def reset(self):
        with self._lock:
            for cam_id in self._cam_state:
                self._cam_state[cam_id] = {"state": "unknown", "first_seen": 0.0, "present_since": 0.0, "absent_since": 0.0}
            self._absent_triggered = False

    def get_state(self, camera_id):
        with self._lock:
            return self._cam_state.get(camera_id, {}).get("state", "unknown")

    @property
    def any_present(self):
        with self._lock:
            return any(s["state"] == "present" for s in self._cam_state.values())
