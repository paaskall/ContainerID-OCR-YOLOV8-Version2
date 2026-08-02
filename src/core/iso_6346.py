import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from itertools import combinations

ISO_RE = re.compile(r"^[A-Z]{4}\d{7}$")

LETTER_VALUES = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19,
    "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29,
    "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36, "Y": 37, "Z": 38,
}

CONFUSABLE = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}

INV_CONFUSABLE = {v: k for k, v in CONFUSABLE.items()}

CATEGORY_OCR_MAP: Dict[str, str] = {
    "0": "U",
    "V": "U",
    "I": "J",
    "1": "J",
    "2": "Z",
    "7": "Z",
}

VALID_CATEGORIES = {"U", "J", "Z"}

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGIT = "0123456789"


@dataclass
class IsoResult:
    text: str
    is_valid: bool
    owner_code: Optional[str] = None
    category_id: Optional[str] = None
    serial: Optional[str] = None
    check_digit: Optional[str] = None
    calc_digit: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class RepairResult:
    repaired_text: Optional[str]
    score: float
    edits: int
    reason: str


def normalize(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _char_value(ch: str) -> Optional[int]:
    if ch.isdigit():
        return int(ch)
    return LETTER_VALUES.get(ch)


def calc_check_digit(code10: str) -> Optional[int]:
    if not code10 or len(code10) != 10:
        return None
    total = 0
    for i, ch in enumerate(code10):
        v = _char_value(ch)
        if v is None:
            return None
        total += v * (2 ** i)
    remainder = total % 11
    digit = remainder % 10
    return digit


def validate_iso(container: str) -> IsoResult:
    t = normalize(container)
    if len(t) != 11:
        return IsoResult(text=t, is_valid=False, reason="len_not_11")
    if not ISO_RE.match(t):
        return IsoResult(text=t, is_valid=False, reason="regex_fail")
    owner = t[:3]
    category = t[3:4]
    serial = t[4:10]
    chk = t[10:11]
    if category not in VALID_CATEGORIES:
        return IsoResult(
            text=t,
            is_valid=False,
            owner_code=owner,
            category_id=category,
            serial=serial,
            check_digit=chk,
            reason="bad_category",
        )
    cd = calc_check_digit(t[:10])
    if cd is None:
        return IsoResult(
            text=t,
            is_valid=False,
            owner_code=owner,
            category_id=category,
            serial=serial,
            check_digit=chk,
            reason="calc_fail",
        )
    calc = str(cd)
    ok = chk == calc
    return IsoResult(
        text=t,
        is_valid=ok,
        owner_code=owner,
        category_id=category,
        serial=serial,
        check_digit=chk,
        calc_digit=calc,
        reason=("ok" if ok else "bad_check_digit"),
    )


def _apply_confusable_fix(s: str) -> str:
    s = list(s)
    for idx in range(4, 10):
        ch = s[idx]
        if ch in CONFUSABLE:
            s[idx] = CONFUSABLE[ch]
    return "".join(s)


def _repair_category(s: str) -> str:
    if len(s) < 4:
        return s
    cat_char = s[3]
    if cat_char in VALID_CATEGORIES:
        return s
    if cat_char in CATEGORY_OCR_MAP:
        return s[:3] + CATEGORY_OCR_MAP[cat_char] + s[4:]
    return s


def _count_edits(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def generate_replacements(prefix: str, positions: tuple):
    pools = []
    for pos in positions:
        if pos < 3:
            pools.append(ALPHA.replace(prefix[pos], ""))
        elif pos == 3:
            pools.append([c for c in VALID_CATEGORIES if c != prefix[pos]])
        else:
            pools.append(DIGIT.replace(prefix[pos], ""))
    def backtrack(idx, current):
        if idx == len(positions):
            yield "".join(current)
            return
        pos = positions[idx]
        for ch in pools[idx]:
            temp = list(current)
            temp[pos] = ch
            yield from backtrack(idx + 1, temp)
    yield from backtrack(0, list(prefix))


def prefix_search_multi_edit(raw: str, max_prefix_edits: int = 2):
    if len(raw) != 11:
        return None
    prefix = raw[:10]
    check_digit = raw[10]
    if not re.match(r"^[A-Z0-9]{10}$", prefix):
        return None
    for edits in range(1, max_prefix_edits + 1):
        for positions in combinations(range(10), edits):
            for candidate_prefix in generate_replacements(prefix, positions):
                if not re.match(r"^[A-Z]{4}\d{6}$", candidate_prefix):
                    continue
                cd = calc_check_digit(candidate_prefix)
                if cd is None:
                    continue
                if str(cd) == check_digit:
                    return candidate_prefix + check_digit, edits
    return None

def repair_iso(container: str, max_edits: int = 4, max_prefix_edits: int = 2) -> RepairResult:
    raw = normalize(container)
    if not raw:
        return RepairResult(None, 0.0, 0, "empty")

    v0 = validate_iso(raw)
    if v0.is_valid:
        return RepairResult(raw, 1.0, 0, "already_valid")

    candidates = []

    def test_candidate(candidate_text: str, base_score: float, reason_label: str):
        if len(candidate_text) != 11:
            return
        v = validate_iso(candidate_text)
        if not v.is_valid:
            return
        edits = abs(len(raw) - len(candidate_text)) + _count_edits(
            raw[:11].ljust(11), candidate_text
        )
        if edits > max_edits:
            return
        score = max(0.0, base_score - 0.1 * edits)
        candidates.append(
            RepairResult(
                repaired_text=candidate_text,
                score=score,
                edits=edits,
                reason=reason_label,
            )
        )

    if len(raw) > 11:
        for idxs in combinations(range(len(raw)), len(raw) - 11):
            temp = list(raw)
            for i in sorted(idxs, reverse=True):
                temp.pop(i)
            candidate = "".join(temp)
            test_candidate(candidate, 0.85, "trim_excess")

    if len(raw) < 11:
        missing = 11 - len(raw)
        for positions in combinations(range(12), missing):
            base = list(raw)
            for pos in sorted(positions):
                if pos <= 3:
                    base.insert(pos, "A")
                elif pos == 4:
                    base.insert(pos, "U")
                else:
                    base.insert(pos, "0")
            candidate = "".join(base[:11])
            test_candidate(candidate, 0.80, "pad_missing")

    if len(raw) == 11:
        c_cat = _repair_category(raw)
        test_candidate(c_cat, 0.90, "fix_category")

        c_conf = _apply_confusable_fix(raw)
        test_candidate(c_conf, 0.92, "confusable_fix")

        prefix10 = raw[:10]
        if re.match(r"^[A-Z]{4}\d{6}$", prefix10):
            cd = calc_check_digit(prefix10)
            if cd is not None:
                forced = prefix10 + str(cd)
                test_candidate(forced, 0.85, "fix_check_digit")

        multi = prefix_search_multi_edit(raw, max_prefix_edits=max_prefix_edits)
        if multi:
            candidate_text, edits = multi
            test_candidate(candidate_text, 0.75, f"prefix_multi_edit_{edits}")

    if not candidates:
        return RepairResult(None, 0.0, 0, "no_valid_candidate")

    best = sorted(candidates, key=lambda x: (-x.score, x.edits))[0]
    return best
    raw = normalize(container)
    if not raw:
        return RepairResult(None, 0.0, 0, "empty")
    v0 = validate_iso(raw)
    if v0.is_valid:
        return RepairResult(raw, 1.0, 0, "already_valid")
    if len(raw) != 11:
        return RepairResult(None, 0.0, 0, "len_not_11")
    candidates: List[Tuple[str, str, float]] = []
    c_cat = _repair_category(raw)
    if c_cat != raw:
        candidates.append((c_cat, "fix_category", 0.90))
    c_conf = _apply_confusable_fix(raw)
    if c_conf != raw:
        candidates.append((c_conf, "confusable_fix", 0.92))
    c_both = _apply_confusable_fix(_repair_category(raw))
    if c_both != raw and c_both not in [c_cat, c_conf]:
        candidates.append((c_both, "category_and_confusable", 0.88))
    base_candidates = [
        (raw, "raw"),
        (c_cat, "fix_category"),
        (c_conf, "confusable_fix"),
        (c_both, "category_and_confusable"),
    ]
    for base, base_reason in base_candidates:
        prefix10 = base[:10]
        if re.match(r"^[A-Z]{4}\d{6}$", prefix10):
            cd = calc_check_digit(prefix10)
            if cd is not None:
                forced = prefix10 + str(cd)
                if forced != base:
                    reason_label = (
                        "fix_check_digit"
                        if base_reason == "raw"
                        else f"{base_reason}+fix_check_digit"
                    )
                    candidates.append((forced, reason_label, 0.85))
    best: Optional[RepairResult] = None
    for candidate_text, reason_label, base_score in candidates:
        edits = _count_edits(raw, candidate_text)
        if edits > max_edits:
            continue
        v = validate_iso(candidate_text)
        if not v.is_valid:
            continue
        score = max(0.0, base_score - 0.1 * edits)
        if best is None or score > best.score or (score == best.score and edits < best.edits):
            best = RepairResult(
                repaired_text=candidate_text,
                score=score,
                edits=edits,
                reason=reason_label,
            )
    if best is not None:
        return best
    multi = prefix_search_multi_edit(raw, max_prefix_edits=max_prefix_edits)
    if multi:
        candidate_text, edits = multi
        return RepairResult(
            repaired_text=candidate_text,
            score=max(0.0, 0.8 - 0.15 * edits),
            edits=edits,
            reason=f"prefix_multi_edit_{edits}",
        )
    return RepairResult(None, 0.0, 0, "no_valid_candidate")