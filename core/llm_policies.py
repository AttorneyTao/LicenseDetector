"""Task-specific rules for responses that are safe enough to reuse.

Unknown tasks are deliberately not cached. Bump a policy version when its
validation or interpretation changes; old entries then become unreachable.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


Context = Mapping[str, Any]
Validator = Callable[[str, Context], str | None]


@dataclass(frozen=True)
class TaskPolicy:
    version: int
    ttl_seconds: int
    validate: Validator


def _json_object(response: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        score = float(value)
        return score if math.isfinite(score) and 0.0 <= score <= 1.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _license_analysis(response: str, context: Context) -> str | None:
    value = _json_object(response)
    content = context.get("content")
    if not value or not isinstance(content, str) or not content.strip():
        return None
    licenses = value.get("main_licenses", value.get("licenses"))
    expression = value.get("spdx_expression")
    if (
        not isinstance(licenses, list)
        or not licenses
        or not all(isinstance(item, str) and item.strip() for item in licenses)
        or _confidence(value.get("confidence")) < 0.9
        or not isinstance(expression, str)
        or not expression.strip()
    ):
        return None
    # Caller-specific source URLs are added *after* the model response.
    if "source_url" in value:
        return None
    value.pop("explanations", None)  # explanatory wording need not match
    return _canonical(value)


def _copyright_json(response: str, context: Context) -> str | None:
    value = _json_object(response)
    notice = value.get("copyright_notice") if value else None
    content = context.get("content", "")
    if not isinstance(notice, str) or not notice.strip() or not isinstance(content, str):
        return None
    parts = [part.strip() for part in notice.split(";")]
    if not all(part and "copyright" in part.casefold() and part in content for part in parts):
        return None
    return _canonical(parts)


def _copyright_text(response: str, context: Context) -> str | None:
    content = context.get("content", "")
    parts = [part.strip() for part in response.strip().split(";")]
    if not isinstance(content, str) or not all(
        part and "copyright" in part.casefold() and part in content for part in parts
    ):
        return None
    return _canonical(parts)


def _version(response: str, context: Context) -> str | None:
    value = _json_object(response)
    candidates = context.get("candidates")
    if not value or not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    chosen = value.get("resolved_version")
    default = context.get("default")
    flag = value.get("used_default_branch")
    if not isinstance(chosen, str) or chosen not in candidates or not isinstance(flag, bool):
        return None
    if flag != (chosen == default):
        return None
    return _canonical({"resolved_version": chosen, "used_default_branch": flag})


def _selected_path(field: str) -> Validator:
    def validate(response: str, context: Context) -> str | None:
        value = _json_object(response)
        candidates = context.get("candidates")
        if not value or not isinstance(candidates, (list, tuple)):
            return None
        chosen = value.get(field)
        if not isinstance(chosen, str) or chosen not in candidates:
            return None
        if _confidence(value.get("confidence")) < 0.8:
            return None
        return chosen
    return validate


def _github_url(response: str, context: Context) -> str | None:
    value = _json_object(response)
    if not value or _confidence(value.get("confidence")) < 0.9:
        return None
    url = value.get("github_url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        return None
    if len([p for p in parsed.path.split("/") if p]) != 2:
        return None
    return url.rstrip("/")


def _spdx(response: str, context: Context) -> str | None:
    value = _json_object(response)
    if not value or _confidence(value.get("confidence")) < 0.9:
        return None
    identifier = value.get("spdx_identifier")
    if not isinstance(identifier, str) or not identifier.strip() or identifier == "UNKNOWN":
        return None
    return identifier.strip()


WEEK = 7 * 24 * 60 * 60
POLICIES: dict[str, TaskPolicy] = {
    "license_analysis": TaskPolicy(1, WEEK, _license_analysis),
    "copyright_extract": TaskPolicy(1, WEEK, _copyright_json),
    "copyright_analysis": TaskPolicy(1, WEEK, _copyright_text),
    "crate_version": TaskPolicy(1, 24 * 60 * 60, _version),
    "npm_version": TaskPolicy(1, 24 * 60 * 60, _version),
    "github_version": TaskPolicy(1, 24 * 60 * 60, _version),
    "license_selector": TaskPolicy(1, WEEK, _selected_path("primary_license_path")),
    "font_license_selector": TaskPolicy(1, WEEK, _selected_path("primary_license_path")),
    "component_license_locator": TaskPolicy(1, WEEK, _selected_path("license_directory")),
    "github_url_finder": TaskPolicy(1, 24 * 60 * 60, _github_url),
    "license_standardize": TaskPolicy(1, WEEK, _spdx),
}
