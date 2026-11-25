# AIM Code 39 Extended decoding, ported verbatim from natn2014/Hioki_data_logger's
# main.py (AIM_MAP + decode_barcode + _decode_model_text) for consistency with the
# barcode content produced by the same label-printing/scanning conventions. zbar
# does NOT decode these escape sequences itself (verified empirically: a raw scan
# of a "+A" Code39 payload comes back as the literal two characters "+A", not the
# Full-ASCII-mapped "a") — this post-processing step is required.
AIM_MAP = {
    '/A': ' ',  '/B': '!',  '/C': '"',  '/D': ',',
    '/E': '%',  '/F': '&',  '/G': "'",  '/H': '(',
    '/I': ')',  '/J': '*',  '/K': '+',  '/L': '/',
    '/M': ':',  '/N': ';',  '/O': '<',  '/P': '=',
    '/Q': '>',  '/R': '?',  '/S': '@',  '/T': '[',
    '/U': '\\', '/V': ']',  '/W': '^',  '/X': '_',
    '/Y': '`',  '/Z': '{',
}


def decode_barcode(raw: str) -> str:
    """Decode AIM Code 39 Extended barcode string to a plain model number.

    Strips the leading check-digit character, converts /X escape pairs to their
    real characters, and stops at /D (field separator).
    """
    if not raw:
        return ''
    s = raw[1:]  # strip check-digit / scanner prefix
    result = ''
    i = 0
    while i < len(s):
        if s[i] == '/' and i + 1 < len(s):
            code = s[i:i + 2].upper()
            if code == '/D':
                break
            if code in AIM_MAP:
                result += AIM_MAP[code]
                i += 2
                continue
        result += s[i]
        i += 1
    return result.strip()


def decode_scanned_text(raw: str) -> str:
    """Unified decode for barcode-scanned text (not manual entry).

    Priority:
    1. Dollar-delimited  — PREFIX$[id]MODEL$SUFFIX  (label-printer format)
    2. AIM Code 39 Extended — /X escape sequences   (scanner format)
    3. Plain text — returned as-is after strip
    """
    raw = raw.strip()
    if not raw:
        return ''
    if '$' in raw:
        first = raw.find('$')
        after = raw[first + 1:]
        second = after.find('$')
        return (after[:second] if second != -1 else after).strip()
    # AIM Code 39: presence of /[A-Z] escape pair signals encoded barcode
    if any(raw[i] == '/' and i + 1 < len(raw) and raw[i + 1].isupper()
           for i in range(len(raw))):
        decoded = decode_barcode(raw)
        if decoded:
            return decoded
    return raw


# --- Standard Code 39 Full ASCII (ISO/IEC 16388), printable characters only ---
_FULL_ASCII = {f"+{c}": c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
_FULL_ASCII.update({f"/{c}": ch for c, ch in zip("ABCDEFGHIJKLO", "!\"#$%&'()*+,/")})
_FULL_ASCII.update({"/Z": ":", "%F": ";", "%G": "<", "%H": "=", "%I": ">", "%J": "?",
                    "%K": "[", "%L": "\\", "%M": "]", "%N": "^", "%O": "_", "%P": "{",
                    "%Q": "|", "%R": "}", "%S": "~", "%V": "@", "%W": "`"})
_CODE39_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"


def decode_full_ascii(raw: str) -> str:
    """Code 39 Full ASCII pairs (+A = a, /O = /, %V = @, ...) to text; other
    characters, and pairs that are not printable, stay as they are."""
    out, i = [], 0
    while i < len(raw):
        pair = raw[i:i + 2]
        if pair in _FULL_ASCII:
            out.append(_FULL_ASCII[pair])
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return "".join(out)


_ENCODE = {ch: pair for pair, ch in _FULL_ASCII.items()}


def encode_full_ascii(text: str) -> str:
    """Text as the Code 39 Full ASCII characters a printer encodes it with
    (a = +A, / = /O, ...); the inverse of decode_full_ascii()."""
    return "".join(c if c in _CODE39_CHARS and c not in "$/+%" else _ENCODE.get(c, c) for c in text)


def part_label(part_no):
    """A part number as shown on screen: Code 39 Full ASCII decoded (/H = (,
    /D = $, +A = a, ...). Display only: the job keeps its saved name."""
    return decode_full_ascii(part_no) if part_no else part_no


def _strip_check_char(raw: str) -> str:
    """`raw` without its last character when that is a valid mod 43 check
    character (a scanner set to transmit it), else ''."""
    if len(raw) < 2 or any(c not in _CODE39_CHARS for c in raw):
        return ""
    body = raw[:-1]
    return body if _CODE39_CHARS[sum(_CODE39_CHARS.index(c) for c in body) % 43] == raw[-1] else ""


def scan_readings(raw: str) -> list:
    """Every plausible text of one Code 39 scan, most literal first: as read,
    Full ASCII decoded, the label-printer / Hioki decode, each also without a
    mod 43 check character. A '/' can be literal (1500D/H27/...) or start an
    escape pair, so the reading is settled by which one names a saved job."""
    raw = raw.strip()
    if not raw:
        return []
    bases = [raw]
    stripped = _strip_check_char(raw)
    if stripped:
        bases.append(stripped)
    readings = []
    for base in bases:
        for text in (base, decode_full_ascii(base), decode_scanned_text(base)):
            text = text.strip()
            if text and text not in readings:
                readings.append(text)
    if "$" in raw:  # label-printer PREFIX$MODEL$SUFFIX: explicit, try it first
        readings.insert(0, readings.pop(readings.index(decode_scanned_text(raw))))
    return readings


def match_part_no(raw: str, part_nos) -> str:
    """The saved part number `raw` is: a saved name encoded back to Code 39
    equals the scan, or one reading of the scan names it (same file name, case
    ignored); else the most literal reading ('' for an empty scan)."""
    from .paths import sanitize_part_no

    part_nos = list(part_nos)
    raw = raw.strip()
    bases = [b for b in (raw, _strip_check_char(raw)) if b]
    for part in part_nos:
        if part in bases or encode_full_ascii(part) in bases:
            return part
    known = {sanitize_part_no(p).lower(): p for p in part_nos}
    readings = scan_readings(raw)
    for text in readings:
        part = known.get(sanitize_part_no(text).lower())
        if part is not None:
            return part
    return readings[0] if readings else ""
