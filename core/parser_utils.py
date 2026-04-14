import json


def find_balanced_json_end(text: str, start: int) -> int:
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_prefixed_json(text: str, prefix: str):
    idx = text.find(prefix)
    if idx < 0:
        return None, text

    start = text.find("{", idx)
    if start < 0:
        return None, text

    end = find_balanced_json_end(text, start)
    if end < 0:
        return None, text

    payload = text[start : end + 1]
    try:
        return json.loads(payload), text[:idx].strip()
    except Exception:
        return None, text
