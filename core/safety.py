from pathlib import Path
from typing import Optional


def blocked_by_mode(
    tool_name: str,
    safety_mode: str,
    high_risk_tool_set: set[str],
    require_high_risk_approval: bool,
    high_risk_reason: str,
) -> Optional[str]:
    if safety_mode == "read_only" and tool_name in {"bash", "write_file", "edit_file", "memory_write"}:
        return f"Blocked by SAFETY_MODE={safety_mode}"
    if (require_high_risk_approval or safety_mode != "trusted_local") and tool_name in high_risk_tool_set:
        return high_risk_reason
    return None


def safe_resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    resolved = p.resolve()
    if not resolved.is_relative_to(base_dir):
        raise PermissionError(f"Path outside workspace blocked: {resolved}")
    return resolved


def validate_write_target(path: Path, sensitive_filenames: set[str]) -> Optional[str]:
    name = path.name.lower()
    if name in sensitive_filenames:
        return f"Refusing to write sensitive file: {name}"
    if any(part.startswith(".git") for part in path.parts):
        return "Refusing to write inside .git metadata"
    return None


def command_policy_block_reason(cmd: str, destructive_re, bash_allowed_prefixes: tuple[str, ...]) -> Optional[str]:
    text = (cmd or "").strip()
    if not text:
        return "Command is empty"
    if destructive_re.search(text):
        return "Destructive command pattern detected"
    lower = text.lower()
    if any(token in lower for token in ["| iex", "invoke-expression", "downloadstring(", "curl ", "wget "]):
        return "Dynamic download/execute pattern blocked"
    if bash_allowed_prefixes and not any(lower.startswith(prefix) for prefix in bash_allowed_prefixes):
        return f"Command family not allowed by policy: {text.split()[0]}"
    return None
