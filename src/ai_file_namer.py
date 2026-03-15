"""Windows-friendly GUI tool for AI-powered file and folder renaming."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".mpeg"}
SUPPORTED_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
INVALID_FILENAME_CHARS = r'<>:"/\\|?*'
MAX_WINDOWS_FILENAME_LENGTH = 255


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
    ):
        self.mode = mode
        self.endpoint = endpoint.strip()
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.debug_callback = debug_callback

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
        case_description = "Title Case words" if preferences.capitalization == "title" else "lowercase words"
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
            response = requests.post(self.endpoint, json=payload, timeout=60)
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
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
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
        case_description = "Title Case words" if preferences.capitalization == "title" else "lowercase words"
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
            response = requests.post(self.endpoint, json=payload, timeout=60)
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
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
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
        case_description = "Title Case words" if preferences.capitalization == "title" else "lowercase words"
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
            response = requests.post(self.endpoint, json=payload, timeout=60)
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
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
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


    def suggest_restructure_plan(self, inventory: Dict[str, object], preferences: FilenamePreferences) -> Dict[str, object]:
        """Request a full-tree restructure plan so AI can reason across all subfolders at once."""
        import requests

        style_description = "spaces between words" if preferences.separator == " " else "underscores between words"
        case_description = "Title Case words" if preferences.capitalization == "title" else "lowercase words"
        prompt = (
            "You are organizing a messy folder tree. "
            "Return JSON only with shape: "
            "{\"folders\": [{\"source\": \"relative/current/path\", \"destination\": \"relative/new/path\"}]}. "
            "Include all folders that should move so the whole structure is reorganized from the selected root. "
            "Do not include files. Only folder paths rooted at the selected root folder. "
            "Each destination is the full new relative folder path. "
            f"Use {style_description} and {case_description}. "
            "Do not include markdown. "
            f"Inventory: {json.dumps(inventory)}"
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            self._emit_debug(
                "AI request: suggest_restructure_plan (local)",
                {"endpoint": self.endpoint, "payload": summarize_debug_payload(payload)},
            )
            response = requests.post(self.endpoint, json=payload, timeout=120)
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
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json.get("choices", [])
            content = choices[0]["message"].get("content", "") if choices else ""
            self._emit_debug(
                "AI response: suggest_restructure_plan (remote)",
                {"status_code": response.status_code, "response": summarize_debug_payload(response_json)},
            )

        return extract_json_object(content)


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


def build_folder_inventory(folder: Path, recursive: bool) -> Dict[str, object]:
    """Build a compact folder-only tree representation for whole-root AI planning."""
    folders = collect_subfolders(folder, recursive=recursive)
    folder_paths = [str(subfolder.relative_to(folder)).replace("\\", "/") for subfolder in folders[:400]]
    return {
        "root_folder": folder.name,
        "folder_count": len(folders),
        "folders": folder_paths,
    }


def format_restructure_preview_paths(original_relative: str, target_relative: str) -> Tuple[str, str, str]:
    """Build readable old/new preview strings for restructure rows in the results table."""
    old_path = original_relative.replace("\\", "/").strip("/") or "."
    new_path = target_relative.replace("\\", "/").strip("/") or "."
    transition = f"{old_path} → {new_path}"
    return old_path, new_path, transition


def sanitize_restructure_operations(
    operations: Sequence[Dict[str, object]],
    root: Path,
    preferences: FilenamePreferences,
) -> List[FolderStructureSuggestion]:
    """Validate and sanitize AI folder-move operations into executable suggestions."""
    suggestions: List[FolderStructureSuggestion] = []
    for operation in operations:
        source_raw = str(operation.get("source", "")).strip().replace("\\", "/")
        destination_raw = str(operation.get("destination", "")).strip().replace("\\", "/")
        if not source_raw or not destination_raw:
            continue

        source_rel = source_raw.strip("/")
        source_path = (root / source_rel).resolve()
        try:
            source_path.relative_to(root.resolve())
        except ValueError:
            continue
        if not source_path.exists() or not source_path.is_dir():
            continue

        # destination is the full new relative folder path under the selected root.
        sanitized_destination = sanitize_relative_destination_path(destination_raw, preferences)
        if not sanitized_destination:
            continue

        suggestions.append(
            FolderStructureSuggestion(
                source_path=source_path,
                original_relative=source_rel,
                target_relative=sanitized_destination,
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

        self.folder_var = tk.StringVar()
        self.provider_mode_var = tk.StringVar(value="Local (Ollama /api/generate)")
        self.endpoint_var = tk.StringVar(value="http://localhost:11434/api/generate")
        self.model_var = tk.StringVar(value="llava")
        self.api_key_var = tk.StringVar()
        self.include_date_var = tk.BooleanVar(value=False)
        self.recursive_scan_var = tk.BooleanVar(value=True)
        self.recursive_folder_rename_var = tk.BooleanVar(value=True)
        self.restructure_recursive_var = tk.BooleanVar(value=True)
        self.date_format_var = tk.StringVar(value="%Y-%m-%d")
        # Default to human-readable names as requested: spaces + title case.
        self.word_separator_var = tk.StringVar(value="White spaces ( )")
        self.capitalization_var = tk.StringVar(value="Title Case")
        self.max_filename_length_var = tk.IntVar(value=96)
        self.max_folder_name_length_var = tk.IntVar(value=96)
        self.include_hashtags_var = tk.BooleanVar(value=False)
        self.hashtag_count_var = tk.IntVar(value=3)
        self.dedupe_keep_var = tk.StringVar(value="Keep first match")
        self.status_var = tk.StringVar(value="Choose a folder and generate AI suggestions.")

        self.suggestions: List[FileSuggestion] = []
        self.folder_suggestions: List[FolderSuggestion] = []
        self.folder_structure_suggestions: List[FolderStructureSuggestion] = []
        self.duplicate_file_groups: List[List[Path]] = []
        self.duplicate_folder_groups: List[List[Path]] = []
        self.rename_history: List[Tuple[Path, Path]] = []
        self.folder_rename_history: List[Tuple[Path, Path]] = []
        self.ui_queue: queue.Queue = queue.Queue()
        self.folder_row_paths: Dict[str, Path] = {}
        self.sort_state: Dict[str, bool] = {}
        self.debug_events: List[str] = []
        self.debug_window: Optional[tk.Toplevel] = None
        self.debug_text: Optional[tk.Text] = None

        self._build_ui()
        self.after(80, self._process_ui_queue)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="📂 Folder", width=12).grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse", command=self._select_folder).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Scan + Suggest", command=self._start_scan).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="🪵 AI Debug", command=self._open_debug_window).grid(row=0, column=5, padx=4)

        option_row = ttk.Frame(top)
        option_row.grid(row=0, column=4, sticky="w", padx=(8, 0))
        ttk.Checkbutton(
            option_row,
            text="Recursive files",
            variable=self.recursive_scan_var,
        ).pack(side=tk.LEFT)
        ttk.Label(
            option_row,
            text="Looks inside subfolders",
            foreground="#666",
        ).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="🧠 AI Mode", width=12).grid(row=1, column=0, sticky="w", pady=(10, 0))
        mode_combo = ttk.Combobox(
            top,
            textvariable=self.provider_mode_var,
            values=["Local (Ollama /api/generate)", "Remote (OpenAI-compatible /v1/chat/completions)"],
            state="readonly",
        )
        mode_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=(10, 0))
        mode_combo.bind("<<ComboboxSelected>>", lambda _: self._handle_mode_change())

        ttk.Label(top, text="🌐 Endpoint").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.endpoint_var).grid(row=2, column=1, sticky="ew", padx=8, pady=(10, 0))

        ttk.Label(top, text="🧩 Model").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.model_var).grid(row=3, column=1, sticky="ew", padx=8, pady=(10, 0))

        ttk.Label(top, text="🔐 API Key").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.api_entry = ttk.Entry(top, textvariable=self.api_key_var, show="*")
        self.api_entry.grid(row=4, column=1, sticky="ew", padx=8, pady=(10, 0))

        date_row = ttk.Frame(top)
        date_row.grid(row=1, column=2, columnspan=3, sticky="w", padx=8)
        ttk.Checkbutton(
            date_row,
            text="Include date",
            variable=self.include_date_var,
        ).pack(side=tk.LEFT)
        ttk.Label(date_row, text="Format:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(date_row, width=14, textvariable=self.date_format_var).pack(side=tk.LEFT)
        ttk.Label(
            date_row,
            text="(strftime, e.g. %Y-%m-%d)",
            foreground="#666",
        ).pack(side=tk.LEFT, padx=8)

        naming_row = ttk.Frame(top)
        naming_row.grid(row=4, column=2, columnspan=3, sticky="w", padx=8, pady=(10, 0))
        ttk.Label(naming_row, text="🧰 Separator:").pack(side=tk.LEFT)
        ttk.Combobox(
            naming_row,
            width=18,
            textvariable=self.word_separator_var,
            state="readonly",
            values=["Underscores (_)", "White spaces ( )"],
        ).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(naming_row, text="🔠 Case:").pack(side=tk.LEFT)
        ttk.Combobox(
            naming_row,
            width=12,
            textvariable=self.capitalization_var,
            state="readonly",
            values=["lowercase", "Title Case"],
        ).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(naming_row, text="📏 Max filename length:").pack(side=tk.LEFT)
        ttk.Spinbox(
            naming_row,
            from_=16,
            to=MAX_WINDOWS_FILENAME_LENGTH,
            width=6,
            textvariable=self.max_filename_length_var,
        ).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(naming_row, text="(includes date + extension)", foreground="#666").pack(side=tk.LEFT)

        folder_name_row = ttk.Frame(top)
        folder_name_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(folder_name_row, text="📁 Max folder name length:").pack(side=tk.LEFT)
        ttk.Spinbox(
            folder_name_row,
            from_=8,
            to=MAX_WINDOWS_FILENAME_LENGTH,
            width=6,
            textvariable=self.max_folder_name_length_var,
        ).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(
            folder_name_row,
            text="Used when generating AI folder suggestions.",
            foreground="#666",
        ).pack(side=tk.LEFT)

        hashtag_row = ttk.Frame(top)
        hashtag_row.grid(row=5, column=2, columnspan=3, sticky="w", padx=8, pady=(8, 0))
        ttk.Checkbutton(
            hashtag_row,
            text="#️⃣ Include hashtags",
            variable=self.include_hashtags_var,
        ).pack(side=tk.LEFT)
        ttk.Label(hashtag_row, text="Count:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Spinbox(hashtag_row, from_=1, to=10, width=4, textvariable=self.hashtag_count_var).pack(side=tk.LEFT)
        ttk.Label(
            hashtag_row,
            text="AI and local post-processing keep tags inside your character limit.",
            foreground="#666",
        ).pack(side=tk.LEFT, padx=8)

        folder_row = ttk.Frame(top)
        folder_row.grid(row=2, column=2, columnspan=3, sticky="w", padx=8)
        ttk.Checkbutton(
            folder_row,
            text="Recursive folder categorisation",
            variable=self.recursive_folder_rename_var,
        ).pack(side=tk.LEFT)
        ttk.Button(folder_row, text="🗂️ Suggest Folder Names", command=self._start_folder_suggestions).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Button(folder_row, text="✅ Apply Folder Renames", command=self._apply_folder_renames).pack(side=tk.LEFT)

        restructure_row = ttk.Frame(top)
        restructure_row.grid(row=3, column=2, columnspan=3, sticky="w", padx=8, pady=(10, 0))
        ttk.Checkbutton(
            restructure_row,
            text="Whole-tree restructure (root + subfolders)",
            variable=self.restructure_recursive_var,
            state="disabled",
        ).pack(side=tk.LEFT)
        ttk.Button(
            restructure_row,
            text="🧭 Suggest Folder Restructure",
            command=self._start_folder_restructure_suggestions,
        ).pack(side=tk.LEFT, padx=(10, 4))
        ttk.Button(
            restructure_row,
            text="✅ Apply Folder Restructure",
            command=self._apply_folder_restructure,
        ).pack(side=tk.LEFT)
        ttk.Label(
            restructure_row,
            text="Uses AI to review root + all subfolders and return a full folder-path reorganisation plan.",
            foreground="#666",
        ).pack(side=tk.LEFT, padx=(8, 0))

        dedupe_row = ttk.Frame(top)
        dedupe_row.grid(row=6, column=2, columnspan=3, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(dedupe_row, text="🧹 Deduplicate:").pack(side=tk.LEFT)
        ttk.Combobox(
            dedupe_row,
            width=16,
            textvariable=self.dedupe_keep_var,
            values=["Keep first match", "Keep last match"],
            state="readonly",
        ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(dedupe_row, text="🔁 Find Duplicates", command=self._start_duplicate_scan).pack(side=tk.LEFT)
        ttk.Button(dedupe_row, text="✅ Apply Deduplication", command=self._apply_deduplication).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(dedupe_row, text="🗑️ Remove Empty Folders", command=self._remove_empty_folders).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(dedupe_row, text="(files + folders)", foreground="#666").pack(side=tk.LEFT, padx=8)

        ttk.Label(
            top,
            text="Tip: Column headers are clickable for sorting. Restructure reviews the whole tree before proposing moves.",
            foreground="#666",
        ).grid(row=7, column=2, columnspan=3, sticky="w", padx=8, pady=(10, 0))

        top.columnconfigure(1, weight=1)

        table_wrapper = ttk.Frame(self, padding=(12, 0, 12, 0))
        table_wrapper.pack(fill=tk.BOTH, expand=True)

        cols = ("original", "suggestion", "final", "status")
        self.tree = ttk.Treeview(table_wrapper, columns=cols, show="headings", selectmode="extended")
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
            label="📂 Open Folder in Explorer",
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

    def _build_provider(self) -> AIProvider:
        """Construct provider from current UI values."""
        return AIProvider(
            mode=self.provider_mode_var.get(),
            endpoint=self.endpoint_var.get(),
            model=self.model_var.get(),
            api_key=self.api_key_var.get(),
            debug_callback=self._queue_debug_event,
        )

    def _queue_debug_event(self, message: str) -> None:
        """Queue debug events from worker threads so UI updates stay thread-safe."""
        self.ui_queue.put(("debug", message))

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

    def _handle_mode_change(self) -> None:
        """Adjust defaults and API key availability based on selected provider."""
        mode = self.provider_mode_var.get()
        if mode.startswith("Local"):
            self.endpoint_var.set("http://localhost:11434/api/generate")
            self.model_var.set(self.model_var.get() or "llava")
            self.api_entry.configure(state="disabled")
        else:
            self.endpoint_var.set("https://api.openai.com/v1/chat/completions")
            self.model_var.set(self.model_var.get() or "gpt-4o-mini")
            self.api_entry.configure(state="normal")

    def _select_folder(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.folder_var.set(selected)

    def _start_scan(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid folder.")
            return

        self.tree.delete(*self.tree.get_children())
        self.suggestions.clear()
        self.folder_structure_suggestions.clear()
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
        capitalization = "title" if self.capitalization_var.get() == "Title Case" else "lower"
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
        self.folder_row_paths.clear()
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
        self.folder_row_paths.clear()
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
            inventory = build_folder_inventory(folder, recursive=True)
            plan = provider.suggest_restructure_plan(inventory=inventory, preferences=preferences)
            raw_operations = plan.get("folders", []) if isinstance(plan, dict) else []
            operation_rows = raw_operations if isinstance(raw_operations, list) else []
            sanitized = sanitize_restructure_operations(operation_rows, root=folder, preferences=preferences)
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("status", f"Restructure planning failed: {exc}"))
            return

        self.folder_structure_suggestions = sanitized
        for idx, proposal in enumerate(self.folder_structure_suggestions, start=1):
            self.ui_queue.put(("restructure_add", proposal))
            self.ui_queue.put(
                ("status", f"Restructure suggestion {idx}/{len(self.folder_structure_suggestions)}: {proposal.original_relative}"),
            )

        self.ui_queue.put(
            (
                "status",
                f"Restructure suggestions ready: {len(self.folder_structure_suggestions)} operation(s). Click column headers to sort.",
            )
        )

    def _apply_folder_restructure(self) -> None:
        """Apply AI-planned folder moves for a root-wide reorganization."""
        if not self.folder_structure_suggestions:
            messagebox.showinfo("No restructure suggestions", "Run 'Suggest Folder Restructure' first.")
            return

        if not messagebox.askyesno(
            "Confirm folder restructure",
            f"Apply {len(self.folder_structure_suggestions)} folder move operation(s)?",
        ):
            return

        root = Path(self.folder_var.get()).expanduser().resolve()
        moved_folders = 0
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
            if str(destination).startswith(str(src) + os.sep):
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
                moved_folders += 1
            except OSError:
                skipped_count += 1

        self.folder_structure_suggestions.clear()
        self.status_var.set(
            "Folder restructure complete: "
            f"moved {moved_folders} folder(s), skipped {skipped_count}."
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
        """Show context menu on folder suggestion rows for quick OS folder access."""
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id not in self.folder_row_paths:
            return
        self.tree.selection_set(row_id)
        self.folder_menu.tk_popup(event.x_root, event.y_root)

    def _open_selected_folder_in_explorer(self) -> None:
        """Open the currently selected folder suggestion in the OS file explorer."""
        selected = self.tree.selection()
        if not selected:
            return
        folder_path = self.folder_row_paths.get(selected[0])
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
                self.tree.insert("", tk.END, values=(rec.original_name, rec.suggested_name, rec.final_name, "Ready"))
            elif kind == "error_row":
                original_name, error_text = msg[1], msg[2]
                self.tree.insert("", tk.END, values=(original_name, "", "", f"Error: {error_text}"))
            elif kind == "folder_add":
                folder_rec: FolderSuggestion = msg[1]
                # Reuse the same output table so folder rename previews are visible before apply.
                row_id = self.tree.insert(
                    "",
                    tk.END,
                    values=(folder_rec.original_name, folder_rec.suggested_name, folder_rec.suggested_name, "Folder Ready"),
                )
                # Keep exact paths for right-click Explorer actions.
                self.folder_row_paths[row_id] = folder_rec.path
            elif kind == "restructure_add":
                move_rec: FolderStructureSuggestion = msg[1]
                # Show both old and new structure paths for clearer pre-apply review.
                old_path, new_path, transition = format_restructure_preview_paths(
                    move_rec.original_relative,
                    move_rec.target_relative,
                )
                row_id = self.tree.insert(
                    "",
                    tk.END,
                    values=(old_path, new_path, transition, "Folder Move"),
                )
                self.folder_row_paths[row_id] = move_rec.source_path
            elif kind == "status":
                self.status_var.set(msg[1])
            elif kind == "debug":
                self._append_debug_output(msg[1])

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
