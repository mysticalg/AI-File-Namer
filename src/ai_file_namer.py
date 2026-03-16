"""Windows-friendly GUI tool for AI-powered file and folder renaming."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import queue
import re
import secrets
import subprocess
import sys
import shutil
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".mpeg"}
SUPPORTED_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
INVALID_FILENAME_CHARS = r'<>:"/\\|?*'
MAX_WINDOWS_FILENAME_LENGTH = 255
SETTINGS_FILE_NAME = ".ai_file_namer_settings.json"
DEFAULT_LOCAL_MODE = "Local (Ollama /api/generate)"
DEFAULT_REMOTE_MODE = "Remote (OpenAI-compatible /v1/chat/completions)"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_REMOTE_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_LOCAL_MODEL = "llava"
DEFAULT_REMOTE_MODEL = "gpt-4o-mini"
DEFAULT_AI_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_OAUTH_CLIENT_ID = os.environ.get("OPENAI_OAUTH_CLIENT_ID", "")
DEFAULT_OPENAI_OAUTH_AUTH_URL = "https://auth.openai.com/oauth/authorize"
DEFAULT_OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_OPENAI_OAUTH_SCOPE = "openid profile email offline_access"
DEFAULT_OPENAI_OAUTH_AUDIENCE = os.environ.get("OPENAI_OAUTH_AUDIENCE", "api.openai.com")
DEFAULT_OPENAI_OAUTH_REDIRECT_HOST = "127.0.0.1"
DEFAULT_OPENAI_OAUTH_REDIRECT_PORT = 8765
DEFAULT_OPENAI_OAUTH_REDIRECT_URI = os.environ.get(
    "OPENAI_OAUTH_REDIRECT_URI",
    f"http://{DEFAULT_OPENAI_OAUTH_REDIRECT_HOST}:{DEFAULT_OPENAI_OAUTH_REDIRECT_PORT}/oauth/callback",
)
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
OPENAI_REGISTRATION_URL = "https://platform.openai.com/signup"
OPENAI_OAUTH_APP_SETUP_URL = "https://platform.openai.com/settings/organization/general"


def default_settings_path() -> Path:
    """Return the per-user settings file used to persist UI preferences."""
    return Path.home() / SETTINGS_FILE_NAME


def load_app_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load persisted settings from disk and fail safely on invalid data."""
    settings_path = path or default_settings_path()
    if not settings_path.exists():
        return {}
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_app_settings(settings: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Persist app settings so users keep their preferences between sessions."""
    settings_path = path or default_settings_path()
    try:
        settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        # Ignore persistence failures to keep the UI responsive and non-blocking.
        return


def _coerce_int_setting(value: Any, default: int) -> int:
    """Convert persisted values to int without crashing when settings files are edited manually."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp_ai_timeout_seconds(value: Any) -> int:
    """Clamp timeout settings to a safe, user-adjustable range."""
    parsed = _coerce_int_setting(value, DEFAULT_AI_TIMEOUT_SECONDS)
    return max(30, min(parsed, 3600))


def describe_capitalization_mode(mode: str) -> str:
    """Return human-readable wording for naming-style instructions sent to the model."""
    if mode == "title":
        return "Title Case words"
    if mode == "natural":
        return "natural sentence-style wording"
    return "lowercase words"


def _pkce_code_challenge(verifier: str) -> str:
    """Build an RFC-7636 PKCE code challenge from a verifier string."""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Capture one OAuth callback and pass the query parameters to the flow owner."""

    server_version = "AIOAuthCallback/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        flat_query = {key: values[0] for key, values in query.items() if values}
        self.server.auth_result = flat_query  # type: ignore[attr-defined]

        if "error" in flat_query:
            page = "<h2>OpenAI authorization failed.</h2><p>You can close this window.</p>"
        else:
            page = "<h2>OpenAI connected successfully.</h2><p>You can return to AI File Namer.</p>"

        body = f"<html><body style='font-family:Segoe UI,Arial'>{page}</body></html>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        """Silence local callback HTTP logs to keep stdout clean."""
        return


class OpenAIOAuthClient:
    """Handle OpenAI OAuth login and token exchange for desktop app sessions."""

    def __init__(
        self,
        client_id: str,
        auth_url: str,
        token_url: str,
        scope: str,
        redirect_uri: str,
        audience: str,
    ):
        self.client_id = client_id.strip()
        self.auth_url = auth_url.strip()
        self.token_url = token_url.strip()
        self.scope = scope.strip()
        self.redirect_uri = redirect_uri.strip() or DEFAULT_OPENAI_OAUTH_REDIRECT_URI
        self.audience = audience.strip()

    def authorize(self, timeout_seconds: int = 180) -> Dict[str, Any]:
        """Run OAuth Authorization Code + PKCE flow and return token payload."""
        import requests

        if not self.client_id:
            raise ValueError(
                "OpenAI OAuth Client ID is missing. Add it in AI Provider settings or set OPENAI_OAUTH_CLIENT_ID."
            )
        if not self.client_id.startswith("app_"):
            raise ValueError("OpenAI OAuth Client ID should start with 'app_'.")

        parsed_redirect = urllib.parse.urlparse(self.redirect_uri)
        if parsed_redirect.scheme != "http" or not parsed_redirect.hostname or not parsed_redirect.port:
            raise ValueError(
                "OpenAI redirect URI must be an http://127.0.0.1:<port>/path style local callback URL."
            )
        if parsed_redirect.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("OpenAI redirect URI host must be 127.0.0.1 or localhost for desktop callback flow.")
        redirect_uri = self.redirect_uri
        verifier = secrets.token_urlsafe(64)
        state = secrets.token_urlsafe(24)
        challenge = _pkce_code_challenge(verifier)

        with http.server.ThreadingHTTPServer(
            (parsed_redirect.hostname, parsed_redirect.port),
            _OAuthCallbackHandler,
        ) as callback_server:
            callback_server.timeout = timeout_seconds
            callback_server.auth_result = None  # type: ignore[attr-defined]

            query = urllib.parse.urlencode(
                {
                    "client_id": self.client_id,
                    "response_type": "code",
                    "redirect_uri": redirect_uri,
                    "scope": self.scope,
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    **({"audience": self.audience} if self.audience else {}),
                }
            )
            auth_link = f"{self.auth_url}?{query}"
            webbrowser.open(auth_link, new=1)
            callback_server.handle_request()
            result = getattr(callback_server, "auth_result", None)

        if not isinstance(result, dict):
            raise RuntimeError(
                "Timed out waiting for OpenAI OAuth callback. "
                "This usually means the app Client ID/Redirect URI pair is not configured correctly in OpenAI. "
                f"Current redirect URI: {redirect_uri}"
            )
        if result.get("state") != state:
            raise RuntimeError("OAuth state mismatch detected. Please try again.")
        if "error" in result:
            description = result.get("error_description", result.get("error", "unknown_error"))
            raise RuntimeError(f"OpenAI OAuth error: {description}")

        auth_code = str(result.get("code", "")).strip()
        if not auth_code:
            raise RuntimeError("Authorization succeeded but no authorization code was returned.")

        token_payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
        token_response = requests.post(self.token_url, data=token_payload, timeout=30)
        token_response.raise_for_status()
        parsed = token_response.json()
        if not isinstance(parsed, dict) or not parsed.get("access_token"):
            raise RuntimeError("Token response did not contain an access token.")
        return parsed


@dataclass
class FilenamePreferences:
    """User-configurable naming preferences applied to model output."""

    separator: str
    capitalization: str
    max_filename_length: int
    max_folder_name_length: int
    include_hashtags: bool
    hashtag_count: int


@dataclass
class FileSuggestion:
    """Stores rename suggestion and source file details."""

    path: Path
    original_name: str
    suggested_name: str
    include_date: bool
    date_text: str
    date_separator: str = "_"

    @property
    def final_name(self) -> str:
        """Build final filename with extension and optional date prefix."""
        stem = self.suggested_name
        if self.include_date and self.date_text:
            stem = f"{self.date_text}{self.date_separator}{stem}"
        return f"{stem}{self.path.suffix.lower()}"


@dataclass
class FolderSuggestion:
    """Stores folder rename suggestion for optional recursive folder naming."""

    path: Path
    original_name: str
    suggested_name: str


@dataclass
class FolderStructureSuggestion:
    """Stores a proposed move to build a cleaner directory hierarchy."""

    source_path: Path
    original_relative: str
    target_relative: str
    item_type: str = "folder"


def collect_media_files(folder: Path, recursive: bool) -> List[Path]:
    """Collect supported media files from a folder.

    Uses deterministic sorting so users get stable scan order across runs.
    """
    glob_pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in folder.glob(glob_pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ]
    return sorted(files)


def collect_subfolders(folder: Path, recursive: bool) -> List[Path]:
    """Collect subfolders to optionally rename with AI."""
    if recursive:
        # Deepest-first ordering avoids path invalidation while renaming nested folders.
        return sorted(
            [p for p in folder.glob("**/*") if p.is_dir()],
            key=lambda p: len(p.parts),
            reverse=True,
        )
    return sorted([p for p in folder.iterdir() if p.is_dir()])


class AIProvider:
    """Simple AI provider abstraction supporting local and remote HTTP APIs."""

    def __init__(
        self,
        mode: str,
        endpoint: str,
        model: str,
        api_key: str = "",
        debug_callback: Optional[Callable[[str], None]] = None,
        timeout_seconds: int = DEFAULT_AI_TIMEOUT_SECONDS,
    ):
        self.mode = mode
        self.endpoint = endpoint.strip()
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.debug_callback = debug_callback
        self.timeout_seconds = clamp_ai_timeout_seconds(timeout_seconds)

    def _emit_debug(self, title: str, details: Dict[str, object]) -> None:
        """Send structured AI request/response diagnostics to the UI debug log."""
        if not self.debug_callback:
            return
        self.debug_callback(format_debug_event(title, details))

    def suggest_name(
        self,
        image_bytes: bytes,
        filename_hint: str,
        target_stem_length: int,
        preferences: FilenamePreferences,
    ) -> str:
        """Send image bytes to configured AI and return a safe filename stem."""
        import requests

        style_description = "spaces between words" if preferences.separator == " " else "underscores between words"
        case_description = describe_capitalization_mode(preferences.capitalization)
        hashtag_description = (
            f" Append up to {preferences.hashtag_count} short hashtags at the end."
            if preferences.include_hashtags and preferences.hashtag_count > 0
            else ""
        )
        prompt = (
            "Return only a filename stem for this media. "
            f"Use {style_description} and {case_description}. "
            "No extension, no punctuation, no markdown, no explanation. "
            "Use descriptive wording and try to use the full allowed length. "
            f"Target stem length: {target_stem_length} characters."
            f"{hashtag_description}"
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {
                "model": self.model,
                "prompt": f"{prompt} Hint original file: {filename_hint}",
                "images": [base64.b64encode(image_bytes).decode("utf-8")],
                "stream": False,
            }
            self._emit_debug(
                "AI request: suggest_name (local)",
                {"endpoint": self.endpoint, "payload": summarize_debug_payload(payload)},
            )
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("response", "")
            self._emit_debug(
                "AI response: suggest_name (local)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt} Hint original file: {filename_hint}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                    + base64.b64encode(image_bytes).decode("utf-8")
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 50,
            }
            self._emit_debug(
                "AI request: suggest_name (remote)",
                {
                    "endpoint": self.endpoint,
                    "headers": summarize_debug_headers(headers),
                    "payload": summarize_debug_payload(payload),
                },
            )
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            self._emit_debug(
                "AI response: suggest_name (remote)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )

        stem = sanitize_filename_stem(
            content,
            separator=preferences.separator,
            capitalization=preferences.capitalization,
            max_length=target_stem_length,
        ) or "untitled_media"

        if preferences.include_hashtags and preferences.hashtag_count > 0:
            stem = append_hashtags(
                stem=stem,
                separator=preferences.separator,
                hashtag_count=preferences.hashtag_count,
                max_length=target_stem_length,
            )

        return stem

    def suggest_folder_name(
        self,
        folder_name: str,
        child_entries: Sequence[str],
        preferences: FilenamePreferences,
    ) -> str:
        """Suggest a folder name using the folder's visible content labels."""
        import requests

        listed_items = ", ".join(child_entries[:40]) if child_entries else "empty_folder"
        style_description = "spaces between words" if preferences.separator == " " else "underscores between words"
        case_description = describe_capitalization_mode(preferences.capitalization)
        prompt = (
            "Return only a concise folder name based on these entries. "
            f"Use {style_description} and {case_description}. "
            "No punctuation, no explanation, no markdown. Use 2 to 6 words. "
            f"Current folder name: {folder_name}. Entries: {listed_items}"
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            self._emit_debug(
                "AI request: suggest_folder_name (local)",
                {"endpoint": self.endpoint, "payload": summarize_debug_payload(payload)},
            )
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("response", "")
            self._emit_debug(
                "AI response: suggest_folder_name (local)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 40,
            }
            self._emit_debug(
                "AI request: suggest_folder_name (remote)",
                {
                    "endpoint": self.endpoint,
                    "headers": summarize_debug_headers(headers),
                    "payload": summarize_debug_payload(payload),
                },
            )
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json.get("choices", [])
            content = choices[0]["message"].get("content", "") if choices else ""
            self._emit_debug(
                "AI response: suggest_folder_name (remote)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )

        return (
            sanitize_filename_stem(
                content,
                separator=preferences.separator,
                capitalization=preferences.capitalization,
                max_length=preferences.max_folder_name_length,
            )
            or "untitled_folder"
        )

    def suggest_folder_structure(
        self,
        folder_name: str,
        child_entries: Sequence[str],
        preferences: FilenamePreferences,
    ) -> str:
        """Suggest a logical category path for a folder, like `photos/travel`."""
        import requests

        listed_items = ", ".join(child_entries[:40]) if child_entries else "empty_folder"
        style_description = "spaces between words" if preferences.separator == " " else "underscores between words"
        case_description = describe_capitalization_mode(preferences.capitalization)
        prompt = (
            "Return only a category path for organizing this folder. "
            "Use 1 to 3 path segments separated by '/'. "
            f"Use {style_description} and {case_description}. "
            "No punctuation outside path separators. No explanation. "
            f"Folder name: {folder_name}. Entries: {listed_items}"
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            self._emit_debug(
                "AI request: suggest_folder_structure (local)",
                {"endpoint": self.endpoint, "payload": summarize_debug_payload(payload)},
            )
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("response", "")
            self._emit_debug(
                "AI response: suggest_folder_structure (local)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 40,
            }
            self._emit_debug(
                "AI request: suggest_folder_structure (remote)",
                {
                    "endpoint": self.endpoint,
                    "headers": summarize_debug_headers(headers),
                    "payload": summarize_debug_payload(payload),
                },
            )
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json.get("choices", [])
            content = choices[0]["message"].get("content", "") if choices else ""
            self._emit_debug(
                "AI response: suggest_folder_structure (remote)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )

        return sanitize_category_path(
            raw=content,
            separator=preferences.separator,
            capitalization=preferences.capitalization,
            max_segment_length=preferences.max_folder_name_length,
            max_depth=3,
        )


    def suggest_restructure_plan(
        self,
        inventory: Dict[str, object],
        preferences: FilenamePreferences,
        extra_instruction: str = "",
    ) -> Dict[str, object]:
        """Request a full-tree folder+file restructure plan.

        The planner works from the root inventory in one pass and returns a complete
        path proposal rather than recursively planning each subtree.
        """
        import requests

        style_description = "spaces between words" if preferences.separator == " " else "underscores between words"
        case_description = describe_capitalization_mode(preferences.capitalization)
        prompt = (
            "You are organizing a messy folder tree. "
            "Primary goal: consolidate and simplify the tree so it has fewer, clearer folders. "
            "Use root context (root name, parent name, and full root path) to infer domain and naming intent. "
            "For example, if context suggests Music/Artist/Bach, expand cryptic catalog labels into meaningful work names when confident. "
            "Return JSON only with shape: "
            "{\"operations\": [{\"type\": \"folder\"|\"file\", \"source\": \"relative/path\", \"destination\": \"relative/path\"}], "
            "\"reorganized_structure\": [\"relative/final/folder/path\"], "
            "\"dedupe_files\": true}. "
            "For folder operations: destination is the target parent path (not final folder name). "
            "For file operations: destination may be a target folder path or a full new file path including a renamed filename. "
            "The reorganized_structure list must represent the full final folder tree from the root. "
            "Plan once from the root inventory (do not recursively re-plan each subfolder). "
            "Use shallow logical categories and consolidate duplicates. "
            "Avoid over-nesting and avoid creating one-off folders when an existing shared category works. "
            "Use file extensions to infer media/document categories and improve consistency of file naming. "
            f"Use {style_description} and {case_description}. "
            "Do not include markdown. "
            "Every source folder should appear exactly once in operations (unless intentionally unchanged). "
            "Group semantically similar folders under shared parent categories. "
            "When domain appears to be classical music, prefer composer/work hierarchy over opaque codes when possible (e.g., Artist > Work Group > Full Work Title). "
            "If uncertain, keep existing names rather than inventing facts. "
            f"{extra_instruction} "
            f"Inventory: {json.dumps(inventory)}"
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            self._emit_debug(
                "AI request: suggest_restructure_plan (local)",
                {"endpoint": self.endpoint, "payload": summarize_debug_payload(payload)},
            )
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("response", "")
            self._emit_debug(
                "AI response: suggest_restructure_plan (local)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 900,
            }
            self._emit_debug(
                "AI request: suggest_restructure_plan (remote)",
                {
                    "endpoint": self.endpoint,
                    "headers": summarize_debug_headers(headers),
                    "payload": summarize_debug_payload(payload),
                },
            )
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json.get("choices", [])
            content = choices[0]["message"].get("content", "") if choices else ""
            self._emit_debug(
                "AI response: suggest_restructure_plan (remote)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )

        return extract_restructure_plan(content)


def summarize_debug_payload(payload: object, max_chars: int = 4000) -> str:
    """Serialize payloads compactly for debug display and cap size for UI responsiveness."""
    try:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    except TypeError:
        serialized = str(payload)

    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}... [truncated {len(serialized) - max_chars} chars]"


def summarize_debug_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Redact secrets in debug headers while keeping request context visible."""
    redacted = dict(headers)
    auth_value = redacted.get("Authorization")
    if auth_value:
        redacted["Authorization"] = "Bearer ***redacted***"
    return redacted


def format_debug_event(title: str, details: Dict[str, object]) -> str:
    """Build a timestamped debug event line block for the debug output window."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    body = summarize_debug_payload(details, max_chars=5000)
    return f"[{timestamp}] {title}\n{body}\n\n"


def sanitize_category_path(
    raw: str,
    separator: str,
    capitalization: str,
    max_segment_length: int,
    max_depth: int = 3,
) -> str:
    """Normalize model output into a safe, shallow folder category path."""
    parts = re.split(r"[\\/>|]+", raw)
    cleaned_parts: List[str] = []
    for part in parts:
        cleaned = sanitize_filename_stem(
            part,
            separator=separator,
            capitalization=capitalization,
            max_length=max_segment_length,
        )
        if cleaned:
            cleaned_parts.append(cleaned)
        if len(cleaned_parts) >= max(1, max_depth):
            break
    return "/".join(cleaned_parts)


def sanitize_relative_destination_path(raw: str, preferences: FilenamePreferences, max_depth: int = 6) -> str:
    """Normalize AI destination paths for files/folders and keep them safely relative."""
    cleaned = sanitize_category_path(
        raw=raw,
        separator=preferences.separator,
        capitalization=preferences.capitalization,
        max_segment_length=preferences.max_folder_name_length,
        max_depth=max_depth,
    )
    return cleaned.strip("/")


def extract_json_object(raw: str) -> Dict[str, object]:
    """Extract first JSON object from model output to tolerate extra prose/markdown."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = raw[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_partial_restructure_operations(raw_text: str) -> List[Dict[str, str]]:
    """Recover move operations from partially truncated model output.

    This fallback is used when the model returns incomplete JSON (for example a
    long `operations` list that is cut off before the final closing braces).
    """
    if not raw_text:
        return []

    # Normalize common escaped quoting patterns produced by nested JSON strings.
    normalized = raw_text.replace('\\"', '"').replace('\\n', '\n')
    object_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    operations: List[Dict[str, str]] = []

    for obj_text in object_pattern.findall(normalized):
        type_match = re.search(r'"type"\s*:\s*"([^"]+)"', obj_text)
        if not type_match:
            # Some models return `operation` instead of `type` for single-item payloads.
            type_match = re.search(r'"operation"\s*:\s*"([^"]+)"', obj_text)
        source_match = re.search(r'"source"\s*:\s*"([^"]+)"', obj_text)
        destination_match = re.search(r'"destination"\s*:\s*"([^"]+)"', obj_text)
        if not type_match or not source_match or not destination_match:
            continue
        operation_type = type_match.group(1).strip().lower()
        if operation_type not in {"folder", "file"}:
            continue

        operations.append(
            {
                "type": operation_type,
                "source": source_match.group(1).strip(),
                "destination": destination_match.group(1).strip(),
            }
        )

    return operations


def extract_restructure_plan(raw: object, max_depth: int = 6) -> Dict[str, object]:
    """Extract a usable restructure plan from raw/nested model payloads.

    Some providers (or model responses) wrap the desired plan JSON inside an envelope
    object/string (for example, inside a nested `response` field). This helper walks
    nested values to recover the first object that actually contains `operations`.
    """

    def _walk(value: object, depth: int) -> Optional[Dict[str, object]]:
        if depth <= 0:
            return None

        if isinstance(value, dict):
            operations = value.get("operations")
            if isinstance(operations, list):
                return value
            if isinstance(operations, dict):
                # Some models return a single operation object instead of an array.
                normalized = dict(value)
                normalized["operations"] = [operations]
                return normalized

            # Accept one-operation variants using `operation` + `source` + `destination`.
            op_variant = str(value.get("operation", value.get("type", ""))).strip().lower()
            source = value.get("source")
            destination = value.get("destination")
            if op_variant in {"folder", "file"} and isinstance(source, str) and isinstance(destination, str):
                return {
                    "operations": [
                        {
                            "type": op_variant,
                            "source": source,
                            "destination": destination,
                        }
                    ],
                    "dedupe_files": bool(value.get("dedupe_files", True)),
                    "reorganized_structure": value.get("reorganized_structure", []),
                }

            for key in ("response", "message", "content", "output", "text"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    result = _walk(nested, depth - 1)
                    if result:
                        return result
                elif isinstance(nested, str):
                    result = _walk(nested, depth - 1)
                    if result:
                        return result

                    # Fallback for truncated nested strings that still contain complete operations.
                    partial_ops = extract_partial_restructure_operations(nested)
                    if partial_ops:
                        return {"operations": partial_ops, "dedupe_files": True}
            return None

        if isinstance(value, str):
            parsed = extract_json_object(value)
            if parsed:
                return _walk(parsed, depth - 1)

            # If strict JSON extraction fails, try partial operation recovery.
            partial_ops = extract_partial_restructure_operations(value)
            if partial_ops:
                return {"operations": partial_ops, "dedupe_files": True}

        return None

    return _walk(raw, max_depth) or {}


def parse_ollama_model_names(payload: Dict[str, object]) -> List[str]:
    """Extract sorted unique model names from Ollama `/api/tags` payload."""
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    canonical_by_lower: Dict[str, str] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        raw_name = model.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue

        cleaned_name = raw_name.strip()
        lower_name = cleaned_name.lower()
        if lower_name not in canonical_by_lower:
            canonical_by_lower[lower_name] = cleaned_name

    return [canonical_by_lower[key] for key in sorted(canonical_by_lower)]


def build_ollama_tags_endpoint(generate_endpoint: str) -> str:
    """Convert a configured Ollama generate endpoint into a tags endpoint."""
    base = generate_endpoint.strip() or "http://localhost:11434/api/generate"
    if "/api/" in base:
        prefix = base.split("/api/", 1)[0]
    else:
        prefix = base.rstrip("/")
    return f"{prefix}/api/tags"


def fetch_ollama_model_names(generate_endpoint: str, timeout_seconds: int = 8) -> List[str]:
    """Fetch locally available Ollama model names for dropdown selection."""
    import requests

    tags_endpoint = build_ollama_tags_endpoint(generate_endpoint)
    response = requests.get(tags_endpoint, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    return parse_ollama_model_names(payload)


def build_ollama_missing_guidance(error_text: str) -> str:
    """Return actionable status guidance when local Ollama looks unavailable."""
    normalized = error_text.lower()
    ollama_unreachable_markers = (
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "name or service not known",
        "nodename nor servname provided",
        "connection aborted",
        "winerror 10061",
    )
    if any(marker in normalized for marker in ollama_unreachable_markers):
        return (
            "Could not reach Ollama. If Ollama is not installed, download it at "
            f"{OLLAMA_DOWNLOAD_URL} for local model usage."
        )
    return "Could not fetch Ollama models. You can still type a model manually."


def build_folder_inventory(
    folder: Path,
    recursive: bool,
    max_nodes: int = 240,
    max_files: int = 420,
) -> Dict[str, object]:
    """Build a compact folder+file representation for AI restructure planning.

    The payload intentionally avoids redundant per-row keys so AI requests stay
    small and fast while still preserving enough tree context to plan moves.
    """
    folders = collect_subfolders(folder, recursive=recursive)
    files = sorted(path for path in folder.glob("**/*" if recursive else "*") if path.is_file())
    folder_paths = [str(path.relative_to(folder)).replace("\\", "/") for path in folders]
    sampled_paths = folder_paths[:max_nodes]
    file_paths = [str(path.relative_to(folder)).replace("\\", "/") for path in files]
    sampled_files = file_paths[:max_files]
    direct_children_map: Dict[str, List[str]] = {}
    for subfolder in folders[:max_nodes]:
        rel = str(subfolder.relative_to(folder)).replace("\\", "/")
        direct_folders = [child.name for child in sorted(subfolder.iterdir()) if child.is_dir()]
        # Include only nodes that actually have children to avoid repetitive noise.
        if direct_folders:
            direct_children_map[rel] = direct_folders[:8]

    return {
        "root": folder.name,
        # Include root context so the model can infer domain (e.g., Music/Artists/Bach).
        "root_full_path": str(folder.resolve()),
        "root_parent_name": folder.parent.name,
        # One list of relative folder paths is enough to describe restructure sources.
        "folder_paths": sampled_paths,
        # Include sampled file paths so the planner can suggest tidy file/folder moves too.
        "file_paths": sampled_files,
        # Root-level cardinality helps the model estimate plan scope without row bloat.
        "total_folder_count": len(folders),
        "total_file_count": len(files),
        # Optional child hints preserve local hierarchy without repeating each folder row.
        "direct_children": direct_children_map,
    }


def find_missing_restructure_sources(
    suggestions: Sequence[FolderStructureSuggestion],
    root: Path,
    candidate_folders: Sequence[Path],
) -> List[str]:
    """Return relative folder paths missing from the AI move suggestions."""
    expected = {str(path.relative_to(root)).replace("\\", "/") for path in candidate_folders}
    provided = {item.original_relative.replace("\\", "/") for item in suggestions}
    return sorted(expected - provided)


def format_restructure_preview_paths(original_relative: str, target_relative: str) -> Tuple[str, str, str]:
    """Build readable old/new preview strings for restructure rows in the results table."""
    old_path = original_relative.replace("\\", "/").strip("/") or "."
    new_path = target_relative.replace("\\", "/").strip("/") or "."
    transition = f"{old_path} → {new_path}"
    return old_path, new_path, transition


def normalize_ai_source_relative_path(source_raw: str, root: Path) -> str:
    """Convert AI `source` values into safe paths relative to the selected root.

    Some models return absolute-style paths (for example `/Bach/...`) or include
    the selected root name. We normalize those to plain relative paths.
    """
    cleaned = source_raw.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return ""

    if cleaned == root.name:
        return ""
    if cleaned.startswith(f"{root.name}/"):
        cleaned = cleaned[len(root.name) + 1 :]

    return cleaned.strip("/")


def normalize_ai_destination_relative_path(destination_raw: str, root: Path, preferences: FilenamePreferences) -> str:
    """Convert AI `destination` parent values into safe root-relative paths."""
    cleaned = destination_raw.strip().replace("\\", "/").strip("/")
    if not cleaned or cleaned == root.name:
        return ""
    if cleaned.startswith(f"{root.name}/"):
        cleaned = cleaned[len(root.name) + 1 :]

    return sanitize_relative_destination_path(cleaned, preferences)


def sanitize_restructure_operations(
    operations: Sequence[Dict[str, object]],
    root: Path,
    preferences: FilenamePreferences,
) -> List[FolderStructureSuggestion]:
    """Validate and sanitize AI-proposed operations into executable move suggestions.

    Supports both folder and file operations. Folder destinations are treated as
    parent paths. File destinations may be either parent folders or full file
    targets with a renamed filename.
    """

    def _normalize_segment_for_compare(segment: str) -> str:
        """Create a stable comparison key for path-segment equality checks."""
        return sanitize_filename_stem(
            segment,
            separator=preferences.separator,
            capitalization=preferences.capitalization,
            max_length=preferences.max_folder_name_length,
        ).strip().casefold()

    suggestions: List[FolderStructureSuggestion] = []
    for operation in operations:
        source_raw = str(operation.get("source", "")).strip().replace("\\", "/")
        destination_raw = str(operation.get("destination", "")).strip().replace("\\", "/")
        item_type = str(operation.get("type", "folder")).strip().lower()
        if not source_raw or item_type not in {"folder", "file"}:
            continue

        source_rel = normalize_ai_source_relative_path(source_raw, root)
        if not source_rel:
            continue

        source_path = (root / source_rel).resolve()
        try:
            source_path.relative_to(root.resolve())
        except ValueError:
            continue
        if not source_path.exists():
            continue
        if item_type == "folder" and not source_path.is_dir():
            continue
        if item_type == "file" and not source_path.is_file():
            continue

        # destination is a parent path for folders; for files we also accept a full
        # file path including a renamed leaf name.
        sanitized_destination_parent = normalize_ai_destination_relative_path(
            destination_raw=destination_raw,
            root=root,
            preferences=preferences,
        )
        destination_raw_clean = destination_raw.strip().replace("\\", "/").strip("/")
        if destination_raw_clean.startswith(f"{root.name}/"):
            destination_raw_clean = destination_raw_clean[len(root.name) + 1 :]
        if item_type == "file":
            stem_budget = max(1, preferences.max_filename_length - len(source_path.suffix))
            normalized_stem = sanitize_filename_stem(
                source_path.stem,
                separator=preferences.separator,
                capitalization=preferences.capitalization,
                max_length=stem_budget,
            )
            source_name = f"{normalized_stem or source_path.stem}{source_path.suffix.lower()}"
        else:
            source_name = sanitize_filename_stem(
                source_path.name,
                separator=preferences.separator,
                capitalization=preferences.capitalization,
                max_length=preferences.max_folder_name_length,
            )
            if not source_name:
                source_name = source_path.name

        # Some models return full target paths while others return parent paths.
        # Detect and normalize both styles while preserving safe naming.
        target_relative = source_name
        if sanitized_destination_parent:
            destination_parts = [part for part in Path(sanitized_destination_parent).parts if part not in (".", "..")]
            destination_tail = destination_parts[-1] if destination_parts else ""
            if item_type == "folder" and destination_tail and _normalize_segment_for_compare(destination_tail) == _normalize_segment_for_compare(source_name):
                target_relative = "/".join(destination_parts)
            elif item_type == "file" and destination_raw_clean.lower().endswith(source_path.suffix.lower()):
                # For files, a destination that includes an extension is treated as full file target.
                raw_parts = [part for part in Path(destination_raw_clean).parts if part not in (".", "..")]
                raw_tail = raw_parts[-1] if raw_parts else destination_tail
                normalized_file_name = sanitize_filename_stem(
                    Path(raw_tail).stem,
                    separator=preferences.separator,
                    capitalization=preferences.capitalization,
                    max_length=max(1, preferences.max_filename_length - len(source_path.suffix)),
                )
                if not normalized_file_name:
                    normalized_file_name = source_path.stem
                file_leaf = f"{normalized_file_name}{source_path.suffix.lower()}"
                sanitized_parent_parts = [
                    sanitize_filename_stem(
                        part,
                        separator=preferences.separator,
                        capitalization=preferences.capitalization,
                        max_length=preferences.max_folder_name_length,
                    )
                    or part
                    for part in raw_parts[:-1]
                ]
                target_relative = "/".join([*sanitized_parent_parts, file_leaf])
            else:
                target_relative = "/".join([*destination_parts, source_name])

        # Extra safety net: collapse accidental duplicate trailing segments.
        target_parts = [part for part in Path(target_relative).parts if part not in (".", "..")]
        while (
            item_type == "folder"
            and len(target_parts) >= 2
            and _normalize_segment_for_compare(target_parts[-1]) == _normalize_segment_for_compare(target_parts[-2])
        ):
            target_parts.pop()
        target_relative = "/".join(target_parts) or source_name

        if target_relative.replace("\\", "/").strip("/") == source_rel.replace("\\", "/").strip("/"):
            # Skip no-op moves so the preview table only shows actionable operations.
            continue

        suggestions.append(
            FolderStructureSuggestion(
                source_path=source_path,
                original_relative=source_rel,
                target_relative=target_relative,
                item_type=item_type,
            )
        )
    return suggestions


def sanitize_filename_stem(
    raw: str,
    separator: str = "_",
    capitalization: str = "lower",
    max_length: int = 96,
) -> str:
    """Normalize model response into a Windows-safe filename stem.

    The returned value honors user preferences for separators/capitalization and
    trims to the requested maximum character count.
    """
    if capitalization == "natural":
        # Natural mode preserves model wording/case while still enforcing Windows safety.
        cleaned = raw.strip().replace("\n", " ").replace("\r", " ")
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" _.")
        cleaned = "".join(ch for ch in cleaned if ch not in INVALID_FILENAME_CHARS and ch.isprintable())
        if separator == "_":
            cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = cleaned.strip(" _.")
    else:
        cleaned = raw.strip().lower()
        cleaned = re.sub(r"[`'\"\n\r]", " ", cleaned)
        cleaned = re.sub(r"[^a-z0-9_\-\s]", " ", cleaned)
        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"\s+", separator, cleaned)
        cleaned = cleaned.strip(" _.")
        cleaned = "".join(ch for ch in cleaned if ch not in INVALID_FILENAME_CHARS)
        if capitalization == "title":
            # Preserve chosen separator while capitalizing each token for readability.
            tokens = [token.capitalize() for token in cleaned.split(separator) if token]
            cleaned = separator.join(tokens)
    return cleaned[:max(1, max_length)]


def compute_target_stem_length(
    path: Path,
    include_date: bool,
    date_text: str,
    max_filename_length: int,
    date_separator: str = "_",
) -> int:
    """Compute stem budget so total filename length stays inside the user limit.

    Total length includes stem + optional date prefix + extension.
    """
    extension = path.suffix.lower()
    date_prefix = f"{date_text}{date_separator}" if include_date and date_text else ""
    overhead_length = len(extension) + len(date_prefix)
    safe_total = max(1, min(max_filename_length, MAX_WINDOWS_FILENAME_LENGTH))
    return max(1, safe_total - overhead_length)


def append_hashtags(stem: str, separator: str, hashtag_count: int, max_length: int) -> str:
    """Append deduplicated hashtags derived from stem words while honoring length.

    This helper keeps the app responsive by generating hashtags locally instead of
    requiring an extra model call per file.
    """
    if hashtag_count <= 0:
        return stem[:max_length]

    words = [token.lower() for token in re.split(r"[^a-zA-Z0-9]+", stem) if len(token) >= 3]
    dedup_words: List[str] = []
    for word in words:
        if word not in dedup_words:
            dedup_words.append(word)

    tags = [f"#{word}" for word in dedup_words[:hashtag_count]]
    if not tags:
        return stem[:max_length]

    base = stem.strip()
    if not base:
        base = "untitled"

    # Keep trimming tags from the right until the full filename stem fits.
    while tags:
        combined = f"{base}{separator}{separator.join(tags)}"
        if len(combined) <= max_length:
            return combined
        tags.pop()

    return base[:max_length]


def file_sha256(path: Path) -> str:
    """Calculate SHA256 digest for duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_duplicate_files(files: Sequence[Path]) -> List[List[Path]]:
    """Group duplicate files by content hash."""
    buckets: Dict[str, List[Path]] = {}
    for file_path in files:
        try:
            digest = file_sha256(file_path)
            buckets.setdefault(digest, []).append(file_path)
        except OSError:
            continue
    return [sorted(group) for group in buckets.values() if len(group) > 1]


def folder_signature(folder: Path) -> str:
    """Build a deterministic folder signature from files and contents."""
    digest = hashlib.sha256()
    if not folder.exists() or not folder.is_dir():
        return ""

    for file_path in sorted([p for p in folder.glob("**/*") if p.is_file()]):
        relative = str(file_path.relative_to(folder)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(file_sha256(file_path).encode("utf-8"))
    return digest.hexdigest()


def group_duplicate_folders(folders: Sequence[Path]) -> List[List[Path]]:
    """Group duplicate folders by recursive content signature."""
    buckets: Dict[str, List[Path]] = {}
    for folder in folders:
        signature = folder_signature(folder)
        if signature:
            buckets.setdefault(signature, []).append(folder)
    return [sorted(group) for group in buckets.values() if len(group) > 1]


def remove_empty_folders(folder: Path, recursive: bool = True) -> int:
    """Delete empty subfolders and return how many were removed.

    Traversal is deepest-first so child directories are removed before parents.
    The selected root folder itself is intentionally preserved.
    """
    folders = collect_subfolders(folder, recursive=recursive)
    removed_count = 0
    for candidate in folders:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            candidate.rmdir()
            removed_count += 1
        except OSError:
            # Non-empty or inaccessible folders are skipped to keep cleanup robust.
            continue
    return removed_count


def format_date(pattern: str) -> str:
    """Return formatted date string (defaults to YYYY-MM-DD on invalid pattern)."""
    try:
        return datetime.now().strftime(pattern)
    except ValueError:
        return datetime.now().strftime("%Y-%m-%d")


def extract_video_first_frame(video_path: Path) -> bytes:
    """Extract first frame from video and return JPEG bytes."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read first frame from: {video_path}")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"Unable to encode frame as JPG: {video_path}")
        return buffer.tobytes()
    finally:
        capture.release()


def load_image_bytes(image_path: Path) -> bytes:
    """Open image and transcode to JPEG bytes for consistent AI input."""
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        from io import BytesIO

        out = BytesIO()
        rgb.save(out, format="JPEG", quality=90)
        return out.getvalue()


class App(tk.Tk):
    """Main application window and event logic."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AI File Namer")
        self.geometry("1180x760")
        self.minsize(1000, 650)

        self.settings_path = default_settings_path()
        loaded_settings = load_app_settings(self.settings_path)

        self.folder_var = tk.StringVar(value=str(loaded_settings.get("folder", "")))
        self.provider_mode_var = tk.StringVar(value=str(loaded_settings.get("provider_mode", DEFAULT_LOCAL_MODE)))
        self.endpoint_var = tk.StringVar(value=str(loaded_settings.get("endpoint", DEFAULT_LOCAL_ENDPOINT)))
        self.model_var = tk.StringVar(value=str(loaded_settings.get("model", DEFAULT_LOCAL_MODEL)))
        self.api_key_var = tk.StringVar(value=str(loaded_settings.get("api_key", "")))
        self.openai_client_id_var = tk.StringVar(value=str(loaded_settings.get("openai_client_id", DEFAULT_OPENAI_OAUTH_CLIENT_ID)))
        self.openai_auth_url_var = tk.StringVar(value=str(loaded_settings.get("openai_auth_url", DEFAULT_OPENAI_OAUTH_AUTH_URL)))
        self.openai_token_url_var = tk.StringVar(value=str(loaded_settings.get("openai_token_url", DEFAULT_OPENAI_OAUTH_TOKEN_URL)))
        self.openai_scope_var = tk.StringVar(value=str(loaded_settings.get("openai_scope", DEFAULT_OPENAI_OAUTH_SCOPE)))
        self.openai_audience_var = tk.StringVar(value=str(loaded_settings.get("openai_audience", DEFAULT_OPENAI_OAUTH_AUDIENCE)))
        self.openai_redirect_uri_var = tk.StringVar(value=str(loaded_settings.get("openai_redirect_uri", DEFAULT_OPENAI_OAUTH_REDIRECT_URI)))
        self.include_date_var = tk.BooleanVar(value=bool(loaded_settings.get("include_date", False)))
        self.recursive_scan_var = tk.BooleanVar(value=bool(loaded_settings.get("recursive_scan", True)))
        self.recursive_folder_rename_var = tk.BooleanVar(value=bool(loaded_settings.get("recursive_folder_rename", True)))
        self.restructure_recursive_var = tk.BooleanVar(value=bool(loaded_settings.get("restructure_recursive", True)))
        self.date_format_var = tk.StringVar(value=str(loaded_settings.get("date_format", "%Y-%m-%d")))
        # Default to human-readable names as requested: spaces + title case.
        self.word_separator_var = tk.StringVar(value=str(loaded_settings.get("word_separator", "White spaces ( )")))
        self.capitalization_var = tk.StringVar(value=str(loaded_settings.get("capitalization", "Title Case")))
        self.max_filename_length_var = tk.IntVar(value=_coerce_int_setting(loaded_settings.get("max_filename_length", 96), 96))
        self.max_folder_name_length_var = tk.IntVar(value=_coerce_int_setting(loaded_settings.get("max_folder_name_length", 96), 96))
        self.include_hashtags_var = tk.BooleanVar(value=bool(loaded_settings.get("include_hashtags", False)))
        self.hashtag_count_var = tk.IntVar(value=_coerce_int_setting(loaded_settings.get("hashtag_count", 3), 3))
        self.dedupe_keep_var = tk.StringVar(value=str(loaded_settings.get("dedupe_keep", "Keep first match")))
        self.ai_timeout_seconds_var = tk.IntVar(value=clamp_ai_timeout_seconds(loaded_settings.get("ai_timeout_seconds", DEFAULT_AI_TIMEOUT_SECONDS)))
        self.status_var = tk.StringVar(value="Choose a folder and generate AI suggestions.")
        # Compact card summaries keep active configuration visible on the main window.
        self.provider_card_summary_var = tk.StringVar(value="")
        self.naming_card_summary_var = tk.StringVar(value="")

        # Store per-provider endpoint/model preferences so mode switches don't erase user values.
        self.local_endpoint = str(loaded_settings.get("local_endpoint", DEFAULT_LOCAL_ENDPOINT))
        self.remote_endpoint = str(loaded_settings.get("remote_endpoint", DEFAULT_REMOTE_ENDPOINT))
        self.local_model = str(loaded_settings.get("local_model", DEFAULT_LOCAL_MODEL))
        self.remote_model = str(loaded_settings.get("remote_model", DEFAULT_REMOTE_MODEL))

        self.suggestions: List[FileSuggestion] = []
        self.folder_suggestions: List[FolderSuggestion] = []
        self.folder_structure_suggestions: List[FolderStructureSuggestion] = []
        self.duplicate_file_groups: List[List[Path]] = []
        self.duplicate_folder_groups: List[List[Path]] = []
        self.rename_history: List[Tuple[Path, Path]] = []
        self.folder_rename_history: List[Tuple[Path, Path]] = []
        self.ui_queue: queue.Queue = queue.Queue()
        # Per-row filesystem targets used by the right-click context menu.
        # For file suggestions we store the containing folder; for folder rows we store that folder.
        self.row_open_paths: Dict[str, Path] = {}
        # Keep thumbnail image references alive for Treeview rows (Tkinter drops images if unreferenced).
        self.row_thumbnail_images: Dict[str, tk.PhotoImage] = {}
        self.sort_state: Dict[str, bool] = {}
        self.debug_events: List[str] = []
        self.debug_window: Optional[tk.Toplevel] = None
        self.debug_text: Optional[tk.Text] = None
        self.ollama_models: List[str] = []
        self.oauth_in_progress = False
        # Dialog widget references are optional because settings now live in popup windows.
        self.model_combo: Optional[ttk.Combobox] = None
        self.oauth_connect_button: Optional[ttk.Button] = None
        self.oauth_disconnect_button: Optional[ttk.Button] = None
        self.oauth_status_label: Optional[ttk.Label] = None
        self._settings_ready = False

        self._build_ui()
        self._register_settings_traces()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._settings_ready = True
        self._save_settings()
        self.after(80, self._process_ui_queue)

    def _build_ui(self) -> None:
        """Build a sectioned, task-first layout so scanning, provider setup, and actions are easy to follow."""
        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.X)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        # 1) Source section: choose input folder and run primary scan action.
        source_frame = ttk.LabelFrame(top, text="📂 Source")
        source_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="Folder").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(source_frame, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)
        ttk.Button(source_frame, text="Browse", command=self._select_folder).grid(row=0, column=2, padx=4, pady=8)
        ttk.Button(source_frame, text="Scan + Suggest", command=self._start_scan).grid(row=0, column=3, padx=4, pady=8)
        ttk.Button(source_frame, text="🪵 AI Debug", command=self._open_debug_window).grid(row=0, column=4, padx=(4, 8), pady=8)
        settings_button = ttk.Menubutton(source_frame, text="⚙️ Settings ▾")
        settings_button.grid(row=0, column=5, padx=(0, 8), pady=8)
        self._attach_tooltip(settings_button, "Open AI Provider and Naming Rules settings.")

        # A compact top dropdown keeps the main workflow uncluttered while exposing advanced controls.
        settings_menu = tk.Menu(settings_button, tearoff=0)
        settings_menu.add_command(label="🧠 AI Provider", command=self._open_ai_provider_settings)
        settings_menu.add_command(label="✍️ Naming Rules", command=self._open_naming_rules_settings)
        settings_button.configure(menu=settings_menu)

        ttk.Checkbutton(
            source_frame,
            text="Recursive files",
            variable=self.recursive_scan_var,
        ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
        ttk.Label(
            source_frame,
            text="Scans files in all subfolders.",
            foreground="#666",
        ).grid(row=1, column=2, columnspan=3, sticky="w", pady=(0, 8))

        # 2) Compact settings cards: full configuration now opens from top settings dropdown.
        provider_card = ttk.LabelFrame(top, text="🧠 AI Provider")
        provider_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0), padx=(0, 6))
        provider_card.columnconfigure(0, weight=1)
        ttk.Label(
            provider_card,
            text="Current provider in use:",
            foreground="#666",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(
            provider_card,
            textvariable=self.provider_card_summary_var,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(
            provider_card,
            text="Open ⚙️ Settings ▾ → 🧠 AI Provider to change mode/model.",
            foreground="#666",
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        naming_card = ttk.LabelFrame(top, text="✍️ Naming Rules")
        naming_card.grid(row=1, column=1, sticky="nsew", pady=(10, 0), padx=(6, 0))
        naming_card.columnconfigure(0, weight=1)
        ttk.Label(
            naming_card,
            text="Current naming style:",
            foreground="#666",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(
            naming_card,
            textvariable=self.naming_card_summary_var,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(
            naming_card,
            text="Open ⚙️ Settings ▾ → ✍️ Naming Rules to adjust casing/separators.",
            foreground="#666",
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        # 4) Folder operations section: renaming, restructure, dedupe and cleanup.
        folder_ops_frame = ttk.LabelFrame(top, text="🗂️ Folder Operations")
        folder_ops_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        folder_ops_frame.columnconfigure(4, weight=1)

        ttk.Checkbutton(
            folder_ops_frame,
            text="Recursive folder categorisation",
            variable=self.recursive_folder_rename_var,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Button(folder_ops_frame, text="🗂️ Suggest Folder Names", command=self._start_folder_suggestions).grid(
            row=0, column=1, sticky="w", padx=4, pady=(8, 0)
        )
        ttk.Button(folder_ops_frame, text="✅ Apply Folder Renames", command=self._apply_folder_renames).grid(
            row=0, column=2, sticky="w", padx=4, pady=(8, 0)
        )

        ttk.Checkbutton(
            folder_ops_frame,
            text="Include subfolders in restructure",
            variable=self.restructure_recursive_var,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Button(
            folder_ops_frame,
            text="🧭 Suggest Folder Restructure",
            command=self._start_folder_restructure_suggestions,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Button(
            folder_ops_frame,
            text="✅ Apply Folder Restructure",
            command=self._apply_folder_restructure,
        ).grid(row=1, column=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(
            folder_ops_frame,
            text="Reviews the whole tree and proposes grouped destinations.",
            foreground="#666",
        ).grid(row=1, column=3, columnspan=2, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(folder_ops_frame, text="Deduplicate").grid(row=2, column=0, sticky="w", padx=8, pady=(8, 8))
        ttk.Combobox(
            folder_ops_frame,
            width=16,
            textvariable=self.dedupe_keep_var,
            values=["Keep first match", "Keep last match"],
            state="readonly",
        ).grid(row=2, column=1, sticky="w", padx=4, pady=(8, 8))
        ttk.Button(folder_ops_frame, text="🔁 Find Duplicates", command=self._start_duplicate_scan).grid(
            row=2, column=2, sticky="w", padx=4, pady=(8, 8)
        )
        ttk.Button(folder_ops_frame, text="✅ Apply Deduplication", command=self._apply_deduplication).grid(
            row=2, column=3, sticky="w", padx=4, pady=(8, 8)
        )
        ttk.Button(folder_ops_frame, text="🗑️ Remove Empty Folders", command=self._remove_empty_folders).grid(
            row=2, column=4, sticky="w", padx=4, pady=(8, 8)
        )

        ttk.Label(
            top,
            text="Tip: click table column headers to sort suggestions before applying actions.",
            foreground="#666",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        table_wrapper = ttk.Frame(self, padding=(12, 0, 12, 0))
        table_wrapper.pack(fill=tk.BOTH, expand=True)

        cols = ("original", "suggestion", "final", "status")
        # Use the tree column (#0) to render image thumbnails while still keeping sortable headings.
        self.tree = ttk.Treeview(table_wrapper, columns=cols, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Preview")
        self.tree.column("#0", width=68, minwidth=56, anchor="center", stretch=False)
        # Clickable column headers for quick sorting while reviewing AI suggestions.
        self.tree.heading("original", text="Original", command=lambda: self._sort_tree_by_column("original"))
        self.tree.heading("suggestion", text="AI Suggestion", command=lambda: self._sort_tree_by_column("suggestion"))
        self.tree.heading("final", text="Final Filename", command=lambda: self._sort_tree_by_column("final"))
        self.tree.heading("status", text="Status", command=lambda: self._sort_tree_by_column("status"))
        self.tree.column("original", width=320)
        self.tree.column("suggestion", width=260)
        self.tree.column("final", width=320)
        self.tree.column("status", width=120, anchor="center")

        yscroll = ttk.Scrollbar(table_wrapper, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        # Right-click menu for folder suggestion rows for quick OS navigation/actions.
        self.folder_menu = tk.Menu(self, tearoff=0)
        self.folder_menu.add_command(
            label="📂 Open Containing Folder",
            command=self._open_selected_folder_in_explorer,
        )
        self.tree.bind("<Button-3>", self._show_row_context_menu)
        yscroll.grid(row=0, column=1, sticky="ns")
        table_wrapper.columnconfigure(0, weight=1)
        table_wrapper.rowconfigure(0, weight=1)

        actions = ttk.Frame(self, padding=12)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="✍️ Rename Selected", command=self._rename_selected).pack(side=tk.LEFT)
        ttk.Button(actions, text="🚀 Rename All", command=self._rename_all).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="↩️ Rollback Last Rename", command=self._rollback).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.status_var, padding=(12, 0, 12, 12), foreground="#005a9c").pack(anchor="w")

        self._handle_mode_change()
        self._refresh_settings_card_summaries()
        self._refresh_oauth_status_label()

    def _capture_provider_fields_for_mode(self) -> None:
        """Store endpoint/model values for the currently selected provider mode."""
        if self.provider_mode_var.get().startswith("Local"):
            self.local_endpoint = self.endpoint_var.get().strip() or DEFAULT_LOCAL_ENDPOINT
            self.local_model = self.model_var.get().strip() or DEFAULT_LOCAL_MODEL
        else:
            self.remote_endpoint = self.endpoint_var.get().strip() or DEFAULT_REMOTE_ENDPOINT
            self.remote_model = self.model_var.get().strip() or DEFAULT_REMOTE_MODEL

    def _settings_snapshot(self) -> Dict[str, Any]:
        """Collect all user-facing controls into one persisted settings object."""
        self._capture_provider_fields_for_mode()
        return {
            "folder": self.folder_var.get(),
            "provider_mode": self.provider_mode_var.get(),
            "endpoint": self.endpoint_var.get(),
            "model": self.model_var.get(),
            "api_key": self.api_key_var.get(),
            "openai_client_id": self.openai_client_id_var.get(),
            "openai_auth_url": self.openai_auth_url_var.get(),
            "openai_token_url": self.openai_token_url_var.get(),
            "openai_scope": self.openai_scope_var.get(),
            "openai_audience": self.openai_audience_var.get(),
            "openai_redirect_uri": self.openai_redirect_uri_var.get(),
            "include_date": self.include_date_var.get(),
            "recursive_scan": self.recursive_scan_var.get(),
            "recursive_folder_rename": self.recursive_folder_rename_var.get(),
            "restructure_recursive": self.restructure_recursive_var.get(),
            "date_format": self.date_format_var.get(),
            "word_separator": self.word_separator_var.get(),
            "capitalization": self.capitalization_var.get(),
            # Guard IntVar reads so transient empty Spinbox edits never crash autosave.
            "max_filename_length": self._safe_int_var_get(self.max_filename_length_var, 96),
            "max_folder_name_length": self._safe_int_var_get(self.max_folder_name_length_var, 96),
            "include_hashtags": self.include_hashtags_var.get(),
            "hashtag_count": self._safe_int_var_get(self.hashtag_count_var, 3),
            "dedupe_keep": self.dedupe_keep_var.get(),
            "ai_timeout_seconds": clamp_ai_timeout_seconds(
                self._safe_int_var_get(self.ai_timeout_seconds_var, DEFAULT_AI_TIMEOUT_SECONDS)
            ),
            "local_endpoint": self.local_endpoint,
            "remote_endpoint": self.remote_endpoint,
            "local_model": self.local_model,
            "remote_model": self.remote_model,
        }

    def _save_settings(self, *_: object) -> None:
        """Persist settings after UI changes so relaunch restores the full workspace state."""
        if not self._settings_ready:
            return
        save_app_settings(self._settings_snapshot(), self.settings_path)
        self._refresh_settings_card_summaries()

    def _refresh_settings_card_summaries(self) -> None:
        """Refresh compact provider/naming summary text shown on the main window cards."""
        mode_text = "Local" if self.provider_mode_var.get().startswith("Local") else "Remote"
        active_model = self.model_var.get().strip() or (DEFAULT_LOCAL_MODEL if mode_text == "Local" else DEFAULT_REMOTE_MODEL)
        self.provider_card_summary_var.set(f"{mode_text} mode • model: {active_model}")

        separator_text = "spaces" if self.word_separator_var.get().startswith("White spaces") else "underscores"
        case_text = self.capitalization_var.get()
        self.naming_card_summary_var.set(f"{case_text} • {separator_text}")

    def _register_settings_traces(self) -> None:
        """Attach variable traces so every control change is automatically persisted."""
        tracked_vars = [
            self.folder_var,
            self.provider_mode_var,
            self.endpoint_var,
            self.model_var,
            self.api_key_var,
            self.openai_client_id_var,
            self.openai_auth_url_var,
            self.openai_token_url_var,
            self.openai_scope_var,
            self.openai_audience_var,
            self.openai_redirect_uri_var,
            self.include_date_var,
            self.recursive_scan_var,
            self.recursive_folder_rename_var,
            self.restructure_recursive_var,
            self.date_format_var,
            self.word_separator_var,
            self.capitalization_var,
            self.max_filename_length_var,
            self.max_folder_name_length_var,
            self.include_hashtags_var,
            self.hashtag_count_var,
            self.dedupe_keep_var,
            self.ai_timeout_seconds_var,
        ]
        for variable in tracked_vars:
            variable.trace_add("write", self._save_settings)

    def _on_close(self) -> None:
        """Save settings and close app."""
        self._save_settings()
        self.destroy()

    def _build_provider(self) -> AIProvider:
        """Construct provider from current UI values."""
        return AIProvider(
            mode=self.provider_mode_var.get(),
            endpoint=self.endpoint_var.get(),
            model=self.model_var.get(),
            api_key=self.api_key_var.get(),
            debug_callback=self._queue_debug_event,
            timeout_seconds=clamp_ai_timeout_seconds(
                self._safe_int_var_get(self.ai_timeout_seconds_var, DEFAULT_AI_TIMEOUT_SECONDS)
            ),
        )

    def _safe_int_var_get(self, variable: tk.IntVar, default: int) -> int:
        """Safely read IntVar values while users are mid-edit in Spinbox fields."""
        try:
            return int(variable.get())
        except (TypeError, ValueError, tk.TclError):
            return default

    def _queue_debug_event(self, message: str) -> None:
        """Queue debug events from worker threads so UI updates stay thread-safe."""
        self.ui_queue.put(("debug", message))

    def _refresh_oauth_status_label(self) -> None:
        """Show whether a reusable OpenAI OAuth token is available in app settings."""
        if not self.oauth_status_label or not self.oauth_status_label.winfo_exists():
            return
        token = self.api_key_var.get().strip()
        if token:
            prefix = token[:8]
            self.oauth_status_label.configure(text=f"Connected (token starts with: {prefix}…)", foreground="#0a6f2b")
        else:
            self.oauth_status_label.configure(text="Not connected", foreground="#666")

    def _clear_openai_token(self) -> None:
        """Forget persisted OAuth token so remote requests no longer authenticate."""
        self.api_key_var.set("")
        self.status_var.set("Cleared stored OpenAI OAuth token.")
        self._refresh_oauth_status_label()

    def _start_openai_oauth_login(self) -> None:
        """Launch the OAuth flow in a worker thread so the Tkinter UI stays responsive."""
        if self.oauth_in_progress:
            return
        if not self.provider_mode_var.get().startswith("Remote"):
            self.status_var.set("Switch to Remote mode to connect OpenAI OAuth.")
            return

        # Keep connect behavior true OAuth-only: no token copy/paste should ever be needed.
        # If Client ID is missing we fail fast with guidance, rather than sending users to
        # unrelated pages that feel like a non-OAuth login flow.
        if not self.openai_client_id_var.get().strip():
            messagebox.showerror(
                "OpenAI OAuth Client ID missing",
                "OpenAI browser OAuth requires a configured Client ID.\n\n"
                "Set OPENAI_OAUTH_CLIENT_ID (recommended) or enter a Client ID in AI Provider settings, "
                "then click Connect OpenAI again.\n\n"
                "After that, sign-in happens in-browser and returns to this app automatically—"
                "no manual token copy/paste is needed.",
            )
            self.status_var.set("OpenAI OAuth Client ID is missing. Configure it, then retry Connect OpenAI.")
            return

        # Preflight validation catches common config mismatch before opening browser.
        auth_host = urllib.parse.urlparse(self.openai_auth_url_var.get().strip()).hostname or ""
        token_host = urllib.parse.urlparse(self.openai_token_url_var.get().strip()).hostname or ""
        if auth_host != "auth.openai.com" or token_host != "auth.openai.com":
            messagebox.showwarning(
                "OAuth endpoint mismatch",
                "OpenAI OAuth endpoints should normally use auth.openai.com.\n\n"
                "If you changed auth/token URLs intentionally for a compatible provider, ignore this warning.",
            )

        self.oauth_in_progress = True
        if self.oauth_connect_button and self.oauth_connect_button.winfo_exists():
            self.oauth_connect_button.configure(state="disabled")
        self.status_var.set("Opening browser for OpenAI OAuth login…")
        worker = threading.Thread(target=self._openai_oauth_login_worker, daemon=True)
        worker.start()

    def _openai_oauth_login_worker(self) -> None:
        """Execute OAuth login and post result back to the main thread queue."""
        try:
            oauth_client = OpenAIOAuthClient(
                client_id=self.openai_client_id_var.get(),
                auth_url=self.openai_auth_url_var.get(),
                token_url=self.openai_token_url_var.get(),
                scope=self.openai_scope_var.get(),
                audience=self.openai_audience_var.get(),
                redirect_uri=self.openai_redirect_uri_var.get(),
            )
            token_payload = oauth_client.authorize()
            access_token = str(token_payload.get("access_token", "")).strip()
            if not access_token:
                raise RuntimeError("OpenAI token response did not include an access token.")
            self.ui_queue.put(("oauth_connected", access_token))
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("oauth_error", str(exc)))
        finally:
            self.ui_queue.put(("oauth_done", None))

    def _open_debug_window(self) -> None:
        """Open a live debug window showing AI request/response payload summaries."""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.deiconify()
            self.debug_window.lift()
            return

        window = tk.Toplevel(self)
        window.title("AI Debug Output")
        window.geometry("980x520")
        window.minsize(760, 380)
        self.debug_window = window

        controls = ttk.Frame(window, padding=8)
        controls.pack(fill=tk.X)
        ttk.Label(
            controls,
            text="📡 Live AI request/response log (payloads are truncated; Authorization is redacted).",
            foreground="#555",
        ).pack(side=tk.LEFT)
        ttk.Button(controls, text="🧹 Clear", command=self._clear_debug_output).pack(side=tk.RIGHT)

        text_frame = ttk.Frame(window, padding=(8, 0, 8, 8))
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.debug_text = tk.Text(text_frame, wrap="word", font=("Consolas", 10))
        yscroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=yscroll.set)
        self.debug_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        for event in self.debug_events:
            self.debug_text.insert(tk.END, event)
        self.debug_text.see(tk.END)

    def _clear_debug_output(self) -> None:
        """Clear debug event history and on-screen debug text."""
        self.debug_events.clear()
        if self.debug_text and self.debug_text.winfo_exists():
            self.debug_text.delete("1.0", tk.END)

    def _append_debug_output(self, message: str) -> None:
        """Append one debug event entry and keep memory bounded for long sessions."""
        self.debug_events.append(message)
        max_events = 500
        if len(self.debug_events) > max_events:
            self.debug_events = self.debug_events[-max_events:]

        if self.debug_text and self.debug_text.winfo_exists():
            self.debug_text.insert(tk.END, message)
            self.debug_text.see(tk.END)

    def _attach_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Attach a lightweight tooltip so settings controls remain self-explanatory."""
        tooltip_window: Optional[tk.Toplevel] = None

        def show_tooltip(_: tk.Event) -> None:
            nonlocal tooltip_window
            if tooltip_window and tooltip_window.winfo_exists():
                return
            x = widget.winfo_rootx() + 14
            y = widget.winfo_rooty() + widget.winfo_height() + 8
            tooltip_window = tk.Toplevel(widget)
            tooltip_window.wm_overrideredirect(True)
            tooltip_window.wm_geometry(f"+{x}+{y}")
            ttk.Label(
                tooltip_window,
                text=text,
                background="#fffbe6",
                foreground="#333",
                relief="solid",
                borderwidth=1,
                padding=(8, 4),
            ).pack()

        def hide_tooltip(_: tk.Event) -> None:
            nonlocal tooltip_window
            if tooltip_window and tooltip_window.winfo_exists():
                tooltip_window.destroy()
            tooltip_window = None

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def _open_ai_provider_settings(self) -> None:
        """Open AI-provider-specific settings in a dedicated popup dialog."""
        window = tk.Toplevel(self)
        window.title("AI Provider Settings")
        window.geometry("760x390")
        window.minsize(700, 340)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Mode").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        mode_combo = ttk.Combobox(
            frame,
            textvariable=self.provider_mode_var,
            values=["Local (Ollama /api/generate)", "Remote (OpenAI-compatible /v1/chat/completions)"],
            state="readonly",
        )
        mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        mode_combo.bind("<<ComboboxSelected>>", lambda _: self._handle_mode_change())

        ttk.Label(frame, text="Endpoint").grid(row=1, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Entry(frame, textvariable=self.endpoint_var).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))

        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w", padx=8, pady=(8, 0))
        model_row = ttk.Frame(frame)
        model_row.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        model_row.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var)
        self.model_combo.grid(row=0, column=0, sticky="ew")
        fetch_models_button = ttk.Button(model_row, text="↻ Fetch Ollama Models", command=self._start_ollama_model_refresh)
        fetch_models_button.grid(row=0, column=1, padx=(6, 0))
        self._attach_tooltip(fetch_models_button, "Refresh local Ollama models from the configured endpoint.")

        ttk.Label(frame, text="OpenAI OAuth Client ID").grid(row=3, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Entry(frame, textvariable=self.openai_client_id_var).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Label(frame, text="OpenAI OAuth Audience").grid(row=4, column=0, sticky="w", padx=8, pady=(8, 0))
        audience_entry = ttk.Entry(frame, textvariable=self.openai_audience_var)
        audience_entry.grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        self._attach_tooltip(audience_entry, "Usually api.openai.com. Leave default unless OpenAI support says otherwise.")
        ttk.Label(frame, text="OpenAI Redirect URI").grid(row=5, column=0, sticky="w", padx=8, pady=(8, 0))
        redirect_entry = ttk.Entry(frame, textvariable=self.openai_redirect_uri_var)
        redirect_entry.grid(row=5, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        self._attach_tooltip(
            redirect_entry,
            "Must match your OpenAI app exactly (e.g. http://127.0.0.1:8765/oauth/callback).",
        )
        ttk.Label(
            frame,
            text="Tip: Set OPENAI_OAUTH_CLIENT_ID env var to prefill this automatically.",
            foreground="#666",
        ).grid(row=6, column=1, sticky="w", padx=(0, 8), pady=(4, 0))

        oauth_controls = ttk.Frame(frame)
        oauth_controls.grid(row=7, column=1, sticky="w", padx=(0, 8), pady=(8, 0))
        self.oauth_connect_button = ttk.Button(oauth_controls, text="🔐 Connect OpenAI", command=self._start_openai_oauth_login)
        self.oauth_connect_button.pack(side=tk.LEFT)
        self.oauth_disconnect_button = ttk.Button(oauth_controls, text="🧹 Forget Token", command=self._clear_openai_token)
        self.oauth_disconnect_button.pack(side=tk.LEFT, padx=(6, 0))
        oauth_setup_button = ttk.Button(
            oauth_controls,
            text="ℹ️ OAuth Setup Help",
            command=lambda: webbrowser.open(OPENAI_OAUTH_APP_SETUP_URL, new=1),
        )
        oauth_setup_button.pack(side=tk.LEFT, padx=(6, 0))
        self._attach_tooltip(oauth_setup_button, "Open OpenAI docs/settings to create or find your OAuth Client ID.")

        ttk.Label(frame, text="Token Status").grid(row=7, column=0, sticky="w", padx=8, pady=(8, 0))
        self.oauth_status_label = ttk.Label(frame, text="Not connected", foreground="#666")
        self.oauth_status_label.grid(row=8, column=1, sticky="w", padx=(0, 8), pady=(4, 0))

        ttk.Label(frame, text="⏱️ AI timeout (sec)").grid(row=9, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Spinbox(frame, from_=30, to=3600, increment=30, textvariable=self.ai_timeout_seconds_var, width=10).grid(
            row=9, column=1, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        ttk.Label(
            frame,
            text="Tip: Connect OpenAI launches browser OAuth and returns here automatically (no token copy/paste).",
            foreground="#666",
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 8))

        self._handle_mode_change()
        self._refresh_oauth_status_label()

    def _open_naming_rules_settings(self) -> None:
        """Open filename and folder naming preferences in a focused popup dialog."""
        window = tk.Toplevel(self)
        window.title("Naming Rules")
        window.geometry("760x360")
        window.minsize(700, 320)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Checkbutton(frame, text="📅 Include date prefix", variable=self.include_date_var).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Label(frame, text="Format").grid(row=0, column=1, sticky="e", padx=(0, 4), pady=(8, 0))
        ttk.Entry(frame, width=14, textvariable=self.date_format_var).grid(row=0, column=2, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Label(frame, text="(strftime e.g. %Y-%m-%d)", foreground="#666").grid(
            row=0, column=3, sticky="w", padx=(0, 8), pady=(8, 0)
        )

        ttk.Label(frame, text="Separator").grid(row=1, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Combobox(
            frame,
            width=18,
            textvariable=self.word_separator_var,
            state="readonly",
            values=["Underscores (_)", "White spaces ( )"],
        ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Label(frame, text="Case").grid(row=1, column=2, sticky="e", padx=(0, 4), pady=(8, 0))
        ttk.Combobox(
            frame,
            width=12,
            textvariable=self.capitalization_var,
            state="readonly",
            values=["lowercase", "Title Case", "Natural language"],
        ).grid(row=1, column=3, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Label(
            frame,
            text="Natural language keeps AI sentence-style wording (incl. apostrophes).",
            foreground="#666",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

        ttk.Label(frame, text="Max filename length").grid(row=3, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Spinbox(frame, from_=16, to=MAX_WINDOWS_FILENAME_LENGTH, width=6, textvariable=self.max_filename_length_var).grid(
            row=3, column=1, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        ttk.Label(frame, text="Includes date + extension", foreground="#666").grid(
            row=3, column=2, columnspan=2, sticky="w", padx=(0, 8), pady=(8, 0)
        )

        ttk.Label(frame, text="Max folder name length").grid(row=4, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Spinbox(frame, from_=8, to=MAX_WINDOWS_FILENAME_LENGTH, width=6, textvariable=self.max_folder_name_length_var).grid(
            row=4, column=1, sticky="w", padx=(0, 8), pady=(8, 0)
        )

        ttk.Checkbutton(frame, text="#️⃣ Include hashtags", variable=self.include_hashtags_var).grid(
            row=5, column=0, sticky="w", padx=8, pady=(8, 8)
        )
        ttk.Label(frame, text="Count").grid(row=5, column=1, sticky="e", padx=(0, 4), pady=(8, 8))
        ttk.Spinbox(frame, from_=1, to=10, width=4, textvariable=self.hashtag_count_var).grid(
            row=5, column=2, sticky="w", padx=(0, 8), pady=(8, 8)
        )
        ttk.Label(frame, text="Hashtags stay inside your length limit.", foreground="#666").grid(
            row=5, column=3, sticky="w", padx=(0, 8), pady=(8, 8)
        )

    def _handle_mode_change(self) -> None:
        """Adjust defaults and OAuth controls based on selected provider mode."""
        self._capture_provider_fields_for_mode()
        mode = self.provider_mode_var.get()
        if mode.startswith("Local"):
            # Keep the user's local endpoint/model choices instead of resetting each toggle.
            self.endpoint_var.set(self.local_endpoint or DEFAULT_LOCAL_ENDPOINT)
            self.model_var.set(self.local_model or DEFAULT_LOCAL_MODEL)
            if self.model_combo and self.model_combo.winfo_exists():
                self.model_combo.configure(state="normal")
            if self.oauth_connect_button and self.oauth_connect_button.winfo_exists():
                self.oauth_connect_button.configure(state="disabled")
            if self.oauth_disconnect_button and self.oauth_disconnect_button.winfo_exists():
                self.oauth_disconnect_button.configure(state="disabled")
            self._start_ollama_model_refresh()
        else:
            # Keep the user's remote endpoint/model choices instead of resetting each toggle.
            self.endpoint_var.set(self.remote_endpoint or DEFAULT_REMOTE_ENDPOINT)
            self.model_var.set(self.remote_model or DEFAULT_REMOTE_MODEL)
            if self.model_combo and self.model_combo.winfo_exists():
                self.model_combo.configure(state="normal")
            if self.oauth_connect_button and self.oauth_connect_button.winfo_exists():
                self.oauth_connect_button.configure(state="normal" if not self.oauth_in_progress else "disabled")
            if self.oauth_disconnect_button and self.oauth_disconnect_button.winfo_exists():
                self.oauth_disconnect_button.configure(state="normal")

        self._refresh_oauth_status_label()

    def _start_ollama_model_refresh(self) -> None:
        """Load available Ollama models into the model dropdown without blocking the UI."""
        if not self.provider_mode_var.get().startswith("Local"):
            return

        self.ui_queue.put(("status", "Fetching available Ollama models..."))
        worker = threading.Thread(
            target=self._ollama_model_refresh_worker,
            args=(self.endpoint_var.get(),),
            daemon=True,
        )
        worker.start()

    def _ollama_model_refresh_worker(self, endpoint: str) -> None:
        """Fetch model list and publish UI update events."""
        try:
            models = fetch_ollama_model_names(endpoint)
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("debug", format_debug_event("Ollama model refresh failed", {"error": str(exc)})))
            self.ui_queue.put(("status", build_ollama_missing_guidance(str(exc))))
            return

        self.ui_queue.put(("ollama_models", models))
        if models:
            self.ui_queue.put(("status", f"Loaded {len(models)} Ollama model(s) into the dropdown."))
        else:
            self.ui_queue.put(("status", "No Ollama models found. Pull a model or type one manually."))

    def _select_folder(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.folder_var.set(selected)

    def _start_scan(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return

        if self.provider_mode_var.get().startswith("Remote") and not self.api_key_var.get().strip():
            messagebox.showwarning(
                "OpenAI OAuth required",
                "Connect OpenAI first. Remote usage requires an OpenAI account: "
                f"{OPENAI_REGISTRATION_URL}",
            )
            return

        self.tree.delete(*self.tree.get_children())
        self.suggestions.clear()
        self.folder_structure_suggestions.clear()
        self.row_open_paths.clear()
        self.row_thumbnail_images.clear()
        self.status_var.set("Collecting files and requesting AI suggestions...")

        worker = threading.Thread(
            target=self._scan_worker,
            args=(
                folder,
                self.recursive_scan_var.get(),
                self.include_date_var.get(),
                self.date_format_var.get(),
                self._current_preferences(),
            ),
            daemon=True,
        )
        worker.start()

    def _current_preferences(self) -> FilenamePreferences:
        """Read and normalize naming preferences from the UI."""
        separator = " " if self.word_separator_var.get().startswith("White spaces") else "_"
        capitalization_choice = self.capitalization_var.get()
        if capitalization_choice == "Title Case":
            capitalization = "title"
        elif capitalization_choice == "Natural language":
            capitalization = "natural"
        else:
            capitalization = "lower"
        try:
            chosen_length = int(self.max_filename_length_var.get())
        except (TypeError, tk.TclError, ValueError):
            chosen_length = 96
        max_length = max(1, min(chosen_length, MAX_WINDOWS_FILENAME_LENGTH))
        try:
            chosen_folder_length = int(self.max_folder_name_length_var.get())
        except (TypeError, tk.TclError, ValueError):
            chosen_folder_length = 96
        max_folder_length = max(1, min(chosen_folder_length, MAX_WINDOWS_FILENAME_LENGTH))
        try:
            hashtag_raw = int(self.hashtag_count_var.get())
        except (TypeError, tk.TclError, ValueError):
            hashtag_raw = 3
        hashtag_count = max(1, min(hashtag_raw, 10))
        return FilenamePreferences(
            separator=separator,
            capitalization=capitalization,
            max_filename_length=max_length,
            max_folder_name_length=max_folder_length,
            include_hashtags=self.include_hashtags_var.get(),
            hashtag_count=hashtag_count,
        )

    def _scan_worker(
        self,
        folder: Path,
        recursive_scan: bool,
        include_date: bool,
        date_pattern: str,
        preferences: FilenamePreferences,
    ) -> None:
        files = collect_media_files(folder, recursive_scan)
        if not files:
            scope = "in selected folder and subfolders" if recursive_scan else "in selected folder"
            self.ui_queue.put(("status", f"No image/video files found {scope}."))
            return

        provider = self._build_provider()
        date_text = format_date(date_pattern) if include_date else ""

        for idx, path in enumerate(files, start=1):
            try:
                media_bytes = extract_video_first_frame(path) if path.suffix.lower() in VIDEO_EXTENSIONS else load_image_bytes(path)
                target_stem_length = compute_target_stem_length(
                    path=path,
                    include_date=include_date,
                    date_text=date_text,
                    max_filename_length=preferences.max_filename_length,
                    date_separator=preferences.separator,
                )
                suggestion = provider.suggest_name(media_bytes, path.stem, target_stem_length, preferences)
                rec = FileSuggestion(
                    path=path,
                    original_name=str(path.relative_to(folder)),
                    suggested_name=suggestion,
                    include_date=include_date,
                    date_text=date_text,
                    date_separator=preferences.separator,
                )
                self.ui_queue.put(("add", rec))
                self.ui_queue.put(("status", f"Suggested {idx}/{len(files)}: {rec.original_name}"))
            except Exception as exc:  # noqa: BLE001
                self.ui_queue.put(("error_row", str(path.relative_to(folder)), str(exc)))

        self.ui_queue.put(("status", f"Suggestion complete. {len(files)} file(s) processed."))

    def _start_folder_suggestions(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return

        self.folder_suggestions.clear()
        self.folder_structure_suggestions.clear()
        self.row_open_paths.clear()
        self.row_thumbnail_images.clear()
        # Clear and repurpose the output grid so users can review folder names before applying.
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Analyzing folders and generating AI folder names...")

        worker = threading.Thread(
            target=self._folder_suggestion_worker,
            args=(folder, self.recursive_folder_rename_var.get(), self._current_preferences()),
            daemon=True,
        )
        worker.start()

    def _folder_suggestion_worker(self, folder: Path, recursive_folders: bool, preferences: FilenamePreferences) -> None:
        folders = collect_subfolders(folder, recursive_folders)
        if not folders:
            self.ui_queue.put(("status", "No subfolders found to rename."))
            return

        provider = self._build_provider()
        for idx, subfolder in enumerate(folders, start=1):
            try:
                # Use visible child names as a lightweight semantic summary for the model.
                child_entries = sorted([child.name for child in subfolder.iterdir()])
                suggestion = provider.suggest_folder_name(subfolder.name, child_entries, preferences)
                record = FolderSuggestion(
                    path=subfolder,
                    original_name=str(subfolder.relative_to(folder)),
                    suggested_name=suggestion,
                )
                self.folder_suggestions.append(record)
                self.ui_queue.put(("folder_add", record))
                self.ui_queue.put(
                    (
                        "status",
                        f"Folder suggestion {idx}/{len(folders)}: {subfolder.name} -> {suggestion}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.ui_queue.put(("status", f"Folder suggestion failed for {subfolder.name}: {exc}"))

        self.ui_queue.put(("status", f"Folder suggestions ready: {len(self.folder_suggestions)} folder(s)."))

    def _apply_folder_renames(self) -> None:
        if not self.folder_suggestions:
            messagebox.showinfo(
                "No folder suggestions",
                "Run 'Suggest Folder Names' first to prepare AI folder rename suggestions.",
            )
            return

        if not messagebox.askyesno(
            "Confirm folder rename",
            f"Rename {len(self.folder_suggestions)} folder(s) using AI suggestions?",
        ):
            return

        history: List[Tuple[Path, Path]] = []
        separator = self._current_preferences().separator
        for item in self.folder_suggestions:
            src = item.path
            if not src.exists():
                continue

            dst = src.with_name(item.suggested_name)
            counter = 1
            while dst.exists() and dst != src:
                dst = src.with_name(f"{item.suggested_name}{separator}{counter}")
                counter += 1

            if dst == src:
                continue

            try:
                src.rename(dst)
                history.append((src, dst))
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Folder rename failed", f"Failed for {src.name}: {exc}")

        self.folder_rename_history = history
        self.folder_suggestions.clear()
        self.status_var.set(f"Folder rename complete: {len(history)} folder(s) renamed.")

    def _start_folder_restructure_suggestions(self) -> None:
        """Build AI suggestions for moving folders into cleaner category paths."""
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return

        self.folder_structure_suggestions.clear()
        self.row_open_paths.clear()
        self.row_thumbnail_images.clear()
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Analyzing the full folder tree and building an AI restructure plan...")

        worker = threading.Thread(
            target=self._folder_restructure_worker,
            args=(folder, self.restructure_recursive_var.get(), self._current_preferences()),
            daemon=True,
        )
        worker.start()

    def _folder_restructure_worker(self, folder: Path, recursive_folders: bool, preferences: FilenamePreferences) -> None:
        """Generate a whole-tree AI restructure plan and sanitize it into executable moves."""
        candidate_folders = collect_subfolders(folder, recursive_folders)
        if not candidate_folders:
            self.ui_queue.put(("status", "No subfolders found to restructure."))
            return

        provider = self._build_provider()
        try:
            # Large local-model prompts can return empty `response` when context gets too big.
            # Retry with progressively smaller inventories to keep planning reliable on big trees.
            inventory_variants = [(240, 420), (140, 240), (80, 120)]
            plan: Dict[str, object] = {}
            operation_rows: List[Dict[str, object]] = []
            sanitized: List[FolderStructureSuggestion] = []
            inventory = build_folder_inventory(folder, recursive=recursive_folders)

            for variant_index, (node_limit, file_limit) in enumerate(inventory_variants, start=1):
                inventory = build_folder_inventory(
                    folder,
                    recursive=recursive_folders,
                    max_nodes=node_limit,
                    max_files=file_limit,
                )
                plan = provider.suggest_restructure_plan(inventory=inventory, preferences=preferences)
                raw_operations = plan.get("operations", []) if isinstance(plan, dict) else []
                operation_rows = raw_operations if isinstance(raw_operations, list) else []
                sanitized = sanitize_restructure_operations(operation_rows, root=folder, preferences=preferences)
                if operation_rows:
                    if not sanitized:
                        self.ui_queue.put(
                            (
                                "status",
                                "AI returned operations, but none were actionable after sanitization. "
                                "Try rerunning with a different model or adjust naming settings.",
                            )
                        )
                    break
                if variant_index < len(inventory_variants):
                    self.ui_queue.put(
                        (
                            "status",
                            "AI returned no operations; retrying with a more compact tree summary...",
                        )
                    )

            missing_sources = find_missing_restructure_sources(
                suggestions=sanitized,
                root=folder,
                candidate_folders=candidate_folders,
            )

            # Retry once with targeted guidance if the first plan does not cover the full tree.
            if missing_sources:
                retry_hint = (
                    "The previous response missed these source folders: "
                    f"{json.dumps(missing_sources[:120])}. "
                    "Return one complete plan that covers the entire folder tree in a single response, "
                    "while still prioritizing consolidation into fewer, clearer categories."
                )
                plan = provider.suggest_restructure_plan(
                    inventory=inventory,
                    preferences=preferences,
                    extra_instruction=retry_hint,
                )
                raw_operations = plan.get("operations", []) if isinstance(plan, dict) else []
                operation_rows = raw_operations if isinstance(raw_operations, list) else []
                sanitized = sanitize_restructure_operations(operation_rows, root=folder, preferences=preferences)
                missing_sources = find_missing_restructure_sources(
                    suggestions=sanitized,
                    root=folder,
                    candidate_folders=candidate_folders,
                )
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("status", f"Restructure planning failed: {exc}"))
            return

        if missing_sources:
            self.ui_queue.put(
                (
                    "status",
                    "AI plan is partial. "
                    f"Covered {len(candidate_folders) - len(missing_sources)}/{len(candidate_folders)} folders. "
                    "Review suggestions and rerun if needed.",
                )
            )

        self.folder_structure_suggestions = sanitized
        for idx, proposal in enumerate(self.folder_structure_suggestions, start=1):
            self.ui_queue.put(("restructure_add", proposal))
            self.ui_queue.put(
                ("status", f"Restructure suggestion {idx}/{len(self.folder_structure_suggestions)}: {proposal.original_relative}"),
            )

        dedupe_requested = bool(plan.get("dedupe_files", True)) if isinstance(plan, dict) else True
        if dedupe_requested:
            self.ui_queue.put(("status", "AI plan requests duplicate-file cleanup after restructure."))

        self.ui_queue.put(
            (
                "status",
                f"Restructure suggestions ready: {len(self.folder_structure_suggestions)} operation(s). Click column headers to sort.",
            )
        )

    def _apply_folder_restructure(self) -> None:
        """Apply AI-planned moves for folders/files, then remove duplicate files for cleanup."""
        if not self.folder_structure_suggestions:
            messagebox.showinfo("No restructure suggestions", "Run 'Suggest Folder Restructure' first.")
            return

        if not messagebox.askyesno(
            "Confirm folder restructure",
            f"Apply {len(self.folder_structure_suggestions)} move operation(s) and clean duplicate files?",
        ):
            return

        root = Path(self.folder_var.get()).expanduser().resolve()
        moved_folders = 0
        moved_files = 0
        skipped_count = 0

        # Apply deepest items first so parent moves do not invalidate child paths mid-run.
        ordered_moves = sorted(
            self.folder_structure_suggestions,
            key=lambda item: len(item.source_path.parts),
            reverse=True,
        )

        for proposal in ordered_moves:
            src = proposal.source_path
            if not src.exists():
                skipped_count += 1
                continue

            relative_parts = [part for part in Path(proposal.target_relative).parts if part not in (".", "..")]
            if not relative_parts:
                skipped_count += 1
                continue

            destination = root.joinpath(*relative_parts)
            if destination == src:
                continue
            if proposal.item_type == "folder" and str(destination).startswith(str(src) + os.sep):
                skipped_count += 1
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            candidate_destination = destination
            counter = 1
            while candidate_destination.exists() and candidate_destination != src:
                candidate_destination = destination.with_name(f"{destination.name}_{counter}")
                counter += 1

            try:
                src.rename(candidate_destination)
                if proposal.item_type == "file":
                    moved_files += 1
                else:
                    moved_folders += 1
            except OSError:
                skipped_count += 1

        # Final tidy-up: remove exact duplicate files by content hash across the selected tree.
        all_files = [path for path in root.glob("**/*") if path.is_file()]
        duplicate_groups = group_duplicate_files(all_files)
        duplicate_removed = 0
        for group in duplicate_groups:
            for duplicate in sorted(group)[1:]:
                try:
                    duplicate.unlink()
                    duplicate_removed += 1
                except OSError:
                    continue

        self.folder_structure_suggestions.clear()
        self.status_var.set(
            "Folder restructure complete: "
            f"moved {moved_folders} folder(s), moved {moved_files} file(s), "
            f"removed {duplicate_removed} duplicate file(s), skipped {skipped_count}."
        )

    def _start_duplicate_scan(self) -> None:
        """Start duplicate scan in a worker to keep the UI responsive."""
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return

        self.status_var.set("Scanning for duplicate files and folders...")
        worker = threading.Thread(
            target=self._duplicate_scan_worker,
            args=(folder, self.recursive_scan_var.get()),
            daemon=True,
        )
        worker.start()

    def _duplicate_scan_worker(self, folder: Path, recursive_scan: bool) -> None:
        files = collect_media_files(folder, recursive_scan)
        file_groups = group_duplicate_files(files)

        scan_folders = collect_subfolders(folder, recursive=True) if recursive_scan else collect_subfolders(folder, recursive=False)
        folder_groups = group_duplicate_folders(scan_folders)

        self.duplicate_file_groups = file_groups
        self.duplicate_folder_groups = folder_groups
        self.ui_queue.put(
            (
                "status",
                f"Duplicate scan complete: {len(file_groups)} file group(s), {len(folder_groups)} folder group(s).",
            )
        )

    def _apply_deduplication(self) -> None:
        """Apply deduplication by keeping first/last item in each duplicate group."""
        if not self.duplicate_file_groups and not self.duplicate_folder_groups:
            messagebox.showinfo("No duplicate data", "Run 'Find Duplicates' first.")
            return

        if not messagebox.askyesno(
            "Confirm deduplication",
            "This will remove duplicate files and merge duplicate folders. Continue?",
        ):
            return

        keep_last = self.dedupe_keep_var.get() == "Keep last match"
        file_removed = 0
        folder_removed = 0

        for group in self.duplicate_file_groups:
            ordered = sorted(group)
            keeper = ordered[-1] if keep_last else ordered[0]
            for duplicate in ordered:
                if duplicate == keeper or not duplicate.exists():
                    continue
                try:
                    duplicate.unlink()
                    file_removed += 1
                except OSError:
                    continue

        for group in self.duplicate_folder_groups:
            ordered = sorted(group)
            keeper = ordered[-1] if keep_last else ordered[0]
            for duplicate in ordered:
                if duplicate == keeper or not duplicate.exists() or not keeper.exists():
                    continue

                # Merge unique children into keeper, then delete duplicate folder.
                for child in sorted(duplicate.iterdir()):
                    target = keeper / child.name
                    if target.exists():
                        continue
                    try:
                        shutil.move(str(child), str(target))
                    except OSError:
                        continue
                try:
                    duplicate.rmdir()
                    folder_removed += 1
                except OSError:
                    continue

        self.status_var.set(
            f"Deduplication complete: removed {file_removed} duplicate files and {folder_removed} duplicate folders."
        )

    def _show_row_context_menu(self, event: tk.Event) -> None:
        """Show context menu on suggestion rows for quick OS folder access."""
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id not in self.row_open_paths:
            return
        self.tree.selection_set(row_id)
        self.folder_menu.tk_popup(event.x_root, event.y_root)

    def _open_selected_folder_in_explorer(self) -> None:
        """Open the currently selected row's folder in the OS file explorer."""
        selected = self.tree.selection()
        if not selected:
            return
        folder_path = self.row_open_paths.get(selected[0])
        if not folder_path or not folder_path.exists():
            messagebox.showwarning("Folder missing", "The selected folder no longer exists.")
            return

        try:
            if os.name == "nt":
                os.startfile(folder_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder_path)], check=False)
            self.status_var.set(f"Opened folder: {folder_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder failed", f"Could not open folder: {exc}")

    def _build_row_thumbnail(self, path: Path, size: Tuple[int, int] = (44, 44)) -> Optional[tk.PhotoImage]:
        """Build a small preview image for image/video rows in the results table."""
        try:
            from PIL import Image, ImageTk
            from io import BytesIO

            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as image:
                    preview = image.convert("RGB")
                    preview.thumbnail(size)
                    return ImageTk.PhotoImage(preview)

            if path.suffix.lower() in VIDEO_EXTENSIONS:
                # Reuse existing first-frame extraction for consistent lightweight video previews.
                first_frame_jpeg = extract_video_first_frame(path)
                with Image.open(BytesIO(first_frame_jpeg)) as image:
                    preview = image.convert("RGB")
                    preview.thumbnail(size)
                    return ImageTk.PhotoImage(preview)

            return None
        except Exception:
            # Thumbnail generation is best-effort only; skip previews on decode errors.
            return None

    def _remove_empty_folders(self) -> None:
        """Remove empty folders inside the selected path after confirmation."""
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return
        if not messagebox.askyesno(
            "Confirm empty-folder cleanup",
            "Remove all empty folders in the selected directory tree?",
        ):
            return

        removed = remove_empty_folders(folder, recursive=True)
        self.status_var.set(f"Empty-folder cleanup complete: removed {removed} folder(s).")

    def _insert_result_row(self, values: Tuple[str, str, str, str], image: Optional[tk.PhotoImage] = None) -> str:
        """Insert one table row safely and set columns individually.

        Setting column values via `tree.set` avoids Tcl argument edge-cases when AI text
        includes characters that can confuse direct variadic option formatting.
        """
        row_id = self.tree.insert("", tk.END, text="", image=image or "")
        self.tree.set(row_id, "original", values[0])
        self.tree.set(row_id, "suggestion", values[1])
        self.tree.set(row_id, "final", values[2])
        self.tree.set(row_id, "status", values[3])
        return row_id

    def _process_ui_queue(self) -> None:
        while True:
            try:
                msg = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            kind = msg[0]
            if kind == "add":
                rec: FileSuggestion = msg[1]
                self.suggestions.append(rec)
                row_thumbnail = self._build_row_thumbnail(rec.path)
                row_id = self._insert_result_row(
                    values=(rec.original_name, rec.suggested_name, rec.final_name, "Ready"),
                    image=row_thumbnail,
                )
                # File rows should open their containing folder from the context menu.
                self.row_open_paths[row_id] = rec.path.parent
                if row_thumbnail is not None:
                    self.row_thumbnail_images[row_id] = row_thumbnail
            elif kind == "error_row":
                original_name, error_text = msg[1], msg[2]
                self._insert_result_row(values=(original_name, "", "", f"Error: {error_text}"))
            elif kind == "folder_add":
                folder_rec: FolderSuggestion = msg[1]
                # Reuse the same output table so folder rename previews are visible before apply.
                row_id = self._insert_result_row(
                    values=(folder_rec.original_name, folder_rec.suggested_name, folder_rec.suggested_name, "Folder Ready"),
                )
                # Keep exact paths for right-click Explorer actions.
                self.row_open_paths[row_id] = folder_rec.path
            elif kind == "restructure_add":
                move_rec: FolderStructureSuggestion = msg[1]
                # Show both old and new structure paths for clearer pre-apply review.
                old_path, new_path, transition = format_restructure_preview_paths(
                    move_rec.original_relative,
                    move_rec.target_relative,
                )
                row_id = self._insert_result_row(
                    values=(old_path, new_path, transition, f"{move_rec.item_type.title()} Move"),
                )
                # File move previews should open the parent folder; folder previews open that folder directly.
                self.row_open_paths[row_id] = (
                    move_rec.source_path.parent if move_rec.item_type == "file" else move_rec.source_path
                )
            elif kind == "status":
                self.status_var.set(msg[1])
            elif kind == "ollama_models":
                models: List[str] = msg[1]
                self.ollama_models = models
                if self.model_combo and self.model_combo.winfo_exists():
                    self.model_combo.configure(values=models)
                # Keep the current model if valid; otherwise default to the first discovered model.
                if models and self.model_var.get() not in models:
                    self.model_var.set(models[0])
            elif kind == "debug":
                self._append_debug_output(msg[1])
            elif kind == "oauth_connected":
                token = str(msg[1])
                self.api_key_var.set(token)
                self._refresh_oauth_status_label()
                self.status_var.set("OpenAI OAuth connected. Token saved for future runs.")
            elif kind == "oauth_error":
                error_text = str(msg[1])
                if "unknown_error" in error_text:
                    error_text = (
                        f"{error_text}\n\n"
                        "OpenAI returned an unknown_error before callback. Double-check:\n"
                        "• Client ID starts with app_ and is active\n"
                        "• Redirect URI in this app exactly matches OpenAI app settings\n"
                        "• OAuth Audience (if required) matches your OpenAI app configuration\n"
                        "• Auth/Token URLs are auth.openai.com"
                    )
                self.status_var.set(f"OpenAI OAuth failed: {error_text}")
                messagebox.showerror("OpenAI OAuth failed", error_text)
            elif kind == "oauth_done":
                self.oauth_in_progress = False
                self._handle_mode_change()

        self.after(80, self._process_ui_queue)

    def _sort_tree_by_column(self, column_name: str) -> None:
        """Sort current tree rows by the selected column and toggle asc/desc each click."""
        row_ids = list(self.tree.get_children(""))
        if not row_ids:
            return

        # Keep sorting lightweight with case-insensitive string comparison.
        reverse = self.sort_state.get(column_name, False)
        sorted_rows = sorted(row_ids, key=lambda row_id: str(self.tree.set(row_id, column_name)).lower(), reverse=reverse)
        for index, row_id in enumerate(sorted_rows):
            self.tree.move(row_id, "", index)

        self.sort_state[column_name] = not reverse

    def _rename_indices(self, indices: Iterable[int]) -> None:
        rename_pairs: List[Tuple[Path, Path]] = []
        for i in indices:
            suggestion = self.suggestions[i]
            src = suggestion.path
            dst = src.with_name(suggestion.final_name)
            counter = 1
            while dst.exists() and dst != src:
                dst = src.with_name(f"{Path(suggestion.final_name).stem}_{counter}{src.suffix.lower()}")
                counter += 1
            rename_pairs.append((src, dst))

        prompt = f"Rename {len(rename_pairs)} file(s)? This can be rolled back once."
        if not messagebox.askyesno("Confirm rename", prompt):
            return

        history: List[Tuple[Path, Path]] = []
        for src, dst in rename_pairs:
            try:
                src.rename(dst)
                history.append((src, dst))
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Rename failed", f"Failed for {src.name}: {exc}")

        if history:
            self.rename_history = history
            self.status_var.set(f"Renamed {len(history)} file(s). You can rollback last batch.")
            self._start_scan()

    def _rename_selected(self) -> None:
        selected_ids = self.tree.selection()
        if not selected_ids:
            messagebox.showinfo("Nothing selected", "Select one or more rows first.")
            return

        indices = [self.tree.index(item) for item in selected_ids if self.tree.index(item) < len(self.suggestions)]
        if not indices:
            messagebox.showwarning("No valid suggestions", "Selected rows do not contain valid suggestions.")
            return
        self._rename_indices(indices)

    def _rename_all(self) -> None:
        if not self.suggestions:
            messagebox.showinfo("No suggestions", "Run Scan + Suggest first.")
            return
        self._rename_indices(range(len(self.suggestions)))

    def _rollback(self) -> None:
        if not self.rename_history:
            messagebox.showinfo("No rollback data", "No rename batch available for rollback.")
            return

        if not messagebox.askyesno("Confirm rollback", f"Rollback {len(self.rename_history)} file(s)?"):
            return

        rollback_ok = 0
        for original, renamed in reversed(self.rename_history):
            if renamed.exists():
                renamed.rename(original)
                rollback_ok += 1
        self.rename_history.clear()
        self.status_var.set(f"Rollback complete: restored {rollback_ok} file(s).")
        self._start_scan()


def main() -> None:
    """Launch desktop app."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
