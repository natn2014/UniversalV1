"""Automotive glass marking rules: parse the text read off a marking into
fields, recognise which standards it carries, and validate it.

Reference: http://elsie.ua/eng/mark.html. A marking must carry the maker's
trademark and an ECE R43 (or 92/22/EEC) approval; it may also carry DOT
(USA, FMVSS 205 / ANSI Z26.1) and CCC (China) marks, the glass type and a
dot-pattern date code. Pure Python: no Qt, OpenCV or torch, so it can be
unit-tested and reused offline.
"""

import datetime
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

# E-mark country codes (the number inside the circle), as listed on the page.
ECE_COUNTRIES = {
    1: "Germany", 2: "France", 3: "Italy", 4: "Netherlands", 5: "Sweden",
    6: "Belgium", 7: "Hungary", 8: "Czechia", 9: "Spain", 10: "Yugoslavia",
    11: "England", 12: "Austria", 13: "Luxembourg", 14: "Switzerland",
    16: "Norway", 17: "Finland", 18: "Denmark", 19: "Romania", 20: "Poland",
    21: "Portugal", 22: "Russia", 23: "Greece", 24: "Ireland", 25: "Croatia",
    26: "Slovenia", 27: "Slovakia", 28: "Belarus", 29: "Estonia",
    31: "Bosnia and Herzegovina", 32: "Latvia", 37: "Turkey", 42: "EEC",
    43: "Japan",
}

GLASS_TYPES = ("LAMINATED", "TEMPERED")
# Expanded glass type mark; "" (no mark) = light transmission >= 70 %.
GLASS_CLASSES = ("I", "II", "III", "IV", "V", "V-VI")
# ANSI Z26.1 items used on glass: AS1 windshield, AS2 >= 70 %, AS3 < 70 %.
VALID_AS_CLASSES = ("AS1", "AS2", "AS3")

ECE_APPROVAL_DIGITS = (5, 6)
CCC_NUMBER_DIGITS = 6

STANDARD = "standard"
DEFECT = "defect"

# Characters the char model confuses with digits; only applied inside fields
# whose format requires digits.
_DIGITISH = "0-9OIl"
_TO_DIGIT = str.maketrans({"O": "0", "I": "1", "l": "1"})

_ECE_RE = re.compile(rf"43\s*R\s*-?\s*([{_DIGITISH}]++)")
_DOT_RE = re.compile(rf"DOT\s*-?\s*([{_DIGITISH}]++)")
_MODEL_RE = re.compile(rf"(?<![A-Z])M\s*-?\s*([{_DIGITISH}]++)(?![A-Z])")
_AS_RE = re.compile(rf"(?<![A-Z])AS\s*-?\s*([{_DIGITISH}]++)")
_CCC_RE = re.compile(rf"(?<![A-Z0-9])E\s*([{_DIGITISH}]{{{CCC_NUMBER_DIGITS}}})(?![0-9])")
_E_MARK_RE = re.compile(rf"E\s*([{_DIGITISH}]{{1,2}})")


@dataclass
class Issue:
    axis: str  # STANDARD or DEFECT
    code: str  # stable id, e.g. "ECE_COUNTRY_INVALID"
    message: str
    element: Optional[str] = None

    def to_dict(self) -> dict:
        return {"axis": self.axis, "code": self.code,
                "message": self.message, "element": self.element}


@dataclass
class DateCode:
    year: int
    month: int
    day: Optional[int] = None

    def as_date(self) -> datetime.date:
        return datetime.date(self.year, self.month, self.day or 1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}" + (f"-{self.day:02d}" if self.day else "")


@dataclass
class MarkingReading:
    """Fields recognised on one marking; None = not found."""

    manufacturer: Optional[str] = None
    glass_type: Optional[str] = None
    glass_class: Optional[str] = None
    e_country: Optional[int] = None
    ece_approval: Optional[str] = None  # normalised "43R-001582"
    ece_approval_digits: Optional[str] = None
    dot_code: Optional[str] = None  # "682"
    dot_model: Optional[str] = None  # "M3240"
    as_class: Optional[str] = None  # "AS2"
    ccc_mark: bool = False
    ccc_number: Optional[str] = None  # "E000199"
    date_code: Optional[DateCode] = None
    date_code_raw: Optional[List[str]] = None
    unparsed: List[str] = field(default_factory=list)
    parse_issues: List[Issue] = field(default_factory=list)

    @property
    def profile(self) -> List[str]:
        """Standards the marking carries, in a fixed order."""
        found = []
        if self.e_country is not None or self.ece_approval:
            found.append("ECE")
        if self.dot_code:
            found.append("DOT")
        if self.ccc_mark or self.ccc_number:
            found.append("CCC")
        return found

    def describe(self) -> str:
        """One-line 'what type of marking is this' summary."""
        parts = [self.manufacturer or "?", "+".join(self.profile) or "no standard"]
        if self.glass_type:
            parts.append(self.glass_type)
        return " | ".join(parts)

    def fields(self) -> dict:
        return {
            "manufacturer": self.manufacturer,
            "profile": self.profile,
            "glass_type": self.glass_type,
            "glass_class": self.glass_class,
            "e_country": self.e_country,
            "ece_approval": self.ece_approval,
            "dot_code": self.dot_code,
            "dot_model": self.dot_model,
            "as_class": self.as_class,
            "ccc_number": self.ccc_number,
            "date_code": str(self.date_code) if self.date_code else None,
        }


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(text: str) -> str:
    """Uppercase, unify dashes, collapse whitespace."""
    text = text.upper().replace("—", "-").replace("–", "-").replace("_", "-")
    return re.sub(r"\s+", " ", text).strip()


def _digits(raw: str) -> str:
    return raw.translate(_TO_DIGIT)


def _match_word(token: str, words: Sequence[str], max_dist: int = 2) -> Optional[str]:
    best = min(words, key=lambda w: levenshtein(token, w))
    return best if levenshtein(token, best) <= max_dist else None


def parse_e_mark(text: Optional[str]) -> Optional[int]:
    """'E1' / 'E 17' / 'EI7' -> country number; None if unreadable."""
    if not text:
        return None
    m = _E_MARK_RE.search(normalize_text(text))
    return int(_digits(m.group(1))) if m else None


def normalize_glass_class(text: Optional[str]) -> Optional[str]:
    """'//' and 'l l' read off the mark mean roman 'II'."""
    if text is None:
        return None
    t = normalize_text(text).replace("/", "I").replace("L", "I").replace("1", "I")
    return t.replace(" ", "")


def _resolve_year(digit: int, today: datetime.date) -> int:
    """Latest year ending in `digit` that is not after next year."""
    year = today.year - (today.year % 10) + digit
    return year - 10 if year > today.year + 1 else year


def decode_date_code(rows: Sequence[str], rule: str = "page",
                     today: Optional[datetime.date] = None) -> DateCode:
    """Decodes a date code read as text rows ('.' for each dot).

    Single row (Pilkington '3...', Sekurit '...2', Fuyao '9.'): the digit is
    the year's last digit; dots set the month by `rule`:
      - "page":     dots before = n, dots after = n + 6 (the reference
                    page's wording; confirm against real parts)
      - "industry": dots before = 7 - n (6 dots = Jan), dots after = 13 - n
    Three rows (XYG pyramid): top digit = year, sum of row 2 = month, sum of
    row 3 = day.

    Raises ValueError when the rows don't form a valid date code.
    """
    today = today or datetime.date.today()
    rows = [r.strip() for r in rows if r and r.strip()]
    if len(rows) == 1:
        m = re.fullmatch(r"(\.*)\s*([0-9])\s*(\.*)", rows[0].replace(" ", ""))
        if not m:
            raise ValueError(f"unreadable date code {rows[0]!r}")
        before, after = len(m.group(1)), len(m.group(3))
        if (before == 0) == (after == 0):
            raise ValueError("date code needs dots on exactly one side of the digit")
        n = before or after
        if not 1 <= n <= 6:
            raise ValueError(f"{n} dots is outside 1..6")
        if rule == "page":
            month = n if before else n + 6
        elif rule == "industry":
            month = 7 - n if before else 13 - n
        else:
            raise ValueError(f"unknown date rule {rule!r}")
        return DateCode(_resolve_year(int(m.group(2)), today), month)
    if len(rows) == 3:
        try:
            year_digit = int(rows[0])
            month = sum(int(t) for t in rows[1].split())
            day = sum(int(t) for t in rows[2].split())
        except ValueError as exc:
            raise ValueError(f"unreadable XYG date code {rows!r}") from exc
        if not 0 <= year_digit <= 9:
            raise ValueError("XYG year must be one digit")
        if not 1 <= month <= 12:
            raise ValueError(f"month {month} is outside 1..12")
        code = DateCode(_resolve_year(year_digit, today), month, day)
        try:
            code.as_date()
        except ValueError as exc:
            raise ValueError(f"invalid date {code}") from exc
        return code
    raise ValueError(f"date code must have 1 or 3 rows, got {len(rows)}")


def parse(lines: Iterable[str], manufacturer: Optional[str] = None,
          e_mark: Optional[str] = None, ccc_mark: bool = False,
          glass_class: Optional[str] = None,
          date_code: Optional[Sequence[str]] = None,
          date_rule: str = "page",
          today: Optional[datetime.date] = None) -> MarkingReading:
    """Builds a MarkingReading from the marking's text lines and the
    graphical elements the detector found (logo class, E-mark text, CCC
    logo, class mark, date-code rows)."""
    r = MarkingReading(manufacturer=manufacturer, ccc_mark=ccc_mark,
                       e_country=parse_e_mark(e_mark),
                       glass_class=normalize_glass_class(glass_class))
    for raw in lines:
        line = normalize_text(raw)
        if not line:
            continue
        rest = line
        m = _ECE_RE.search(rest)
        if m:
            digits = _digits(m.group(1))
            r.ece_approval_digits = digits
            r.ece_approval = f"43R-{digits}"
            rest = rest.replace(m.group(0), " ")
        m = _DOT_RE.search(rest)
        if m:
            r.dot_code = _digits(m.group(1))
            rest = rest.replace(m.group(0), " ")
        m = _AS_RE.search(rest)
        if m:
            r.as_class = "AS" + _digits(m.group(1))
            rest = rest.replace(m.group(0), " ")
        m = _MODEL_RE.search(rest)
        if m:
            r.dot_model = "M" + _digits(m.group(1))
            rest = rest.replace(m.group(0), " ")
        m = _CCC_RE.search(rest)
        if m:
            r.ccc_number = "E" + _digits(m.group(1))
            rest = rest.replace(m.group(0), " ")
        leftovers = []
        for token in rest.split():
            word = _match_word(token, GLASS_TYPES) if len(token) >= 6 else None
            if word:
                r.glass_type = word
            elif re.fullmatch(r"[/I]{1,3}|IV|V|V-VI", token) and r.glass_class is None:
                r.glass_class = normalize_glass_class(token)
            elif token not in ("-", "SAFETY", "GLASS"):
                leftovers.append(token)
        if leftovers:
            r.unparsed.append(" ".join(leftovers))
    if date_code:
        r.date_code_raw = list(date_code)
        try:
            r.date_code = decode_date_code(date_code, date_rule, today)
        except ValueError as exc:
            r.parse_issues.append(Issue(STANDARD, "DATE_CODE_INVALID", str(exc), "date_code"))
    return r


def _std(code: str, message: str, element: Optional[str] = None) -> Issue:
    return Issue(STANDARD, code, message, element)


def validate_rules(r: MarkingReading, today: Optional[datetime.date] = None) -> List[Issue]:
    """Checks the marking against the standards themselves (no job master)."""
    today = today or datetime.date.today()
    issues = list(r.parse_issues)
    if not r.manufacturer:
        issues.append(_std("MANUFACTURER_MISSING", "No manufacturer trademark", "logo"))
    # ECE R43: the E-mark and the approval number belong together and are mandatory.
    if r.e_country is None:
        issues.append(_std("ECE_MARK_MISSING", "No E-mark (ECE R43 approval)", "e_mark"))
    elif r.e_country not in ECE_COUNTRIES:
        issues.append(_std("ECE_COUNTRY_INVALID", f"E{r.e_country} is not a valid country code", "e_mark"))
    if not r.ece_approval:
        issues.append(_std("ECE_APPROVAL_MISSING", "No 43R approval number", "ece_approval"))
    elif not r.ece_approval_digits.isdigit() or len(r.ece_approval_digits) not in ECE_APPROVAL_DIGITS:
        issues.append(_std("ECE_APPROVAL_FORMAT",
                           f"{r.ece_approval} must have {'/'.join(map(str, ECE_APPROVAL_DIGITS))} digits",
                           "ece_approval"))
    # DOT: FMVSS 205 needs the maker code, AS item and model number together.
    if r.dot_code or r.as_class or r.dot_model:
        if not r.dot_code:
            issues.append(_std("DOT_CODE_MISSING", "AS/M number without DOT code", "dot"))
        if not r.as_class:
            issues.append(_std("AS_CLASS_MISSING", "DOT marking without AS number", "dot"))
        elif r.as_class not in VALID_AS_CLASSES:
            issues.append(_std("AS_CLASS_INVALID", f"{r.as_class} is not one of {', '.join(VALID_AS_CLASSES)}", "dot"))
        if not r.dot_model:
            issues.append(_std("DOT_MODEL_MISSING", "DOT marking without M model number", "dot"))
    if r.as_class == "AS1" and r.glass_type == "TEMPERED":
        issues.append(_std("AS1_NOT_LAMINATED", "AS1 (windshield) glass must be LAMINATED", "dot"))
    # CCC: the logo and the factory number belong together.
    if r.ccc_mark and not r.ccc_number:
        issues.append(_std("CCC_NUMBER_MISSING", "CCC logo without E-number", "ccc"))
    if r.ccc_number and not r.ccc_mark:
        issues.append(_std("CCC_MARK_MISSING", f"{r.ccc_number} without CCC logo", "ccc"))
    if r.glass_class and r.glass_class not in GLASS_CLASSES:
        issues.append(_std("GLASS_CLASS_INVALID", f"Glass class mark {r.glass_class!r} is not valid", "class_mark"))
    if r.date_code and r.date_code.as_date() > today:
        issues.append(_std("DATE_IN_FUTURE", f"Date code {r.date_code} is in the future", "date_code"))
    return issues


# Master keys compared field by field: (master key, reading attribute, element).
_MASTER_FIELDS = (
    ("manufacturer", "manufacturer", "logo"),
    ("glass_type", "glass_type", "glass_type"),
    ("glass_class", "glass_class", "class_mark"),
    ("e_country", "e_country", "e_mark"),
    ("ece_approval", "ece_approval", "ece_approval"),
    ("dot_code", "dot_code", "dot"),
    ("dot_model", "dot_model", "dot"),
    ("as_class", "as_class", "dot"),
    ("ccc_number", "ccc_number", "ccc"),
)


def validate_master(r: MarkingReading, master: dict,
                    today: Optional[datetime.date] = None) -> List[Issue]:
    """Checks the reading against the job's registered master. Master keys
    that are absent or None are not checked."""
    today = today or datetime.date.today()
    issues = []
    expected_profile = master.get("profile")
    if expected_profile is not None and sorted(expected_profile) != sorted(r.profile):
        issues.append(_std("PROFILE_MISMATCH",
                           f"Standards {'+'.join(r.profile) or 'none'} != expected {'+'.join(expected_profile)}"))
    for key, attr, element in _MASTER_FIELDS:
        want = master.get(key)
        if want is None:
            continue
        got = getattr(r, attr)
        if key == "glass_class":
            want = normalize_glass_class(want)
        elif isinstance(want, str):
            want = normalize_text(want)
        if got != want:
            issues.append(_std(f"{key.upper()}_MISMATCH", f"{key}: read {got!r}, expected {want!r}", element))
    max_age = master.get("date_max_age_days")
    if max_age is not None and r.date_code and (today - r.date_code.as_date()).days > max_age:
        issues.append(_std("DATE_TOO_OLD", f"Date code {r.date_code} is older than {max_age} days", "date_code"))
    return issues


def validate(r: MarkingReading, master: Optional[dict] = None,
             today: Optional[datetime.date] = None) -> List[Issue]:
    issues = validate_rules(r, today)
    if master:
        seen = {i.code for i in issues}
        issues += [i for i in validate_master(r, master, today) if i.code not in seen]
    return issues
