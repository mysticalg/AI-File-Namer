"""Windows-friendly GUI tool for AI-powered file renaming."""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".mpeg"}
INVALID_FILENAME_CHARS = r'<>:"/\\|?*'


@dataclass
class FileSuggestion:
    """Stores rename suggestion and source file details."""

    path: Path
    original_name: str
    suggested_name: str
    include_date: bool
    date_text: str

    @property
    def final_name(self) -> str:
        """Build final filename with extension and optional date prefix."""
        stem = self.suggested_name
        if self.include_date and self.date_text:
            stem = f"{self.date_text}_{stem}"
        return f"{stem}{self.path.suffix.lower()}"


class AIProvider:
    """Simple AI provider abstraction supporting local and remote HTTP APIs."""

    def __init__(self, mode: str, endpoint: str, model: str, api_key: str = ""):
        self.mode = mode
        self.endpoint = endpoint.strip()
        self.model = model.strip()
        self.api_key = api_key.strip()

    def suggest_name(self, image_bytes: bytes, filename_hint: str) -> str:
        """Send image bytes to configured AI and return a safe filename stem."""
        import requests

        prompt = (
            "Return only a concise snake_case filename stem for this media. "
            "No extension, punctuation, markdown, or explanation. "
            "Use 3 to 8 words max."
        )

        if self.mode == "Local (Ollama /api/generate)":
            payload = {
                "model": self.model,
                "prompt": f"{prompt} Hint original file: {filename_hint}",
                "images": [base64.b64encode(image_bytes).decode("utf-8")],
                "stream": False,
            }
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("response", "")
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
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            choices = response.json().get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""

        return sanitize_filename_stem(content) or "untitled_media"


def sanitize_filename_stem(raw: str) -> str:
    """Normalize model response into a Windows-safe snake_case filename stem."""
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"[`'\"\n\r]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9_\-\s]", " ", cleaned)
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip(" _.")
    cleaned = "".join(ch for ch in cleaned if ch not in INVALID_FILENAME_CHARS)
    return cleaned[:96]


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
        self.geometry("1160x740")
        self.minsize(980, 620)

        self.folder_var = tk.StringVar()
        self.provider_mode_var = tk.StringVar(value="Local (Ollama /api/generate)")
        self.endpoint_var = tk.StringVar(value="http://localhost:11434/api/generate")
        self.model_var = tk.StringVar(value="llava")
        self.api_key_var = tk.StringVar()
        self.include_date_var = tk.BooleanVar(value=False)
        self.date_format_var = tk.StringVar(value="%Y-%m-%d")
        self.status_var = tk.StringVar(value="Choose a folder and generate AI suggestions.")

        self.suggestions: List[FileSuggestion] = []
        self.rename_history: List[Tuple[Path, Path]] = []
        self.ui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self.after(80, self._process_ui_queue)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="📂 Folder", width=12).grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse", command=self._select_folder).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Scan + Suggest", command=self._start_scan).grid(row=0, column=3, padx=4)

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
        date_row.grid(row=1, column=2, columnspan=2, sticky="w", padx=8)
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

        top.columnconfigure(1, weight=1)

        table_wrapper = ttk.Frame(self, padding=(12, 0, 12, 0))
        table_wrapper.pack(fill=tk.BOTH, expand=True)

        cols = ("original", "suggestion", "final", "status")
        self.tree = ttk.Treeview(table_wrapper, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("original", text="Original")
        self.tree.heading("suggestion", text="AI Suggestion")
        self.tree.heading("final", text="Final Filename")
        self.tree.heading("status", text="Status")
        self.tree.column("original", width=280)
        self.tree.column("suggestion", width=240)
        self.tree.column("final", width=300)
        self.tree.column("status", width=120, anchor="center")

        yscroll = ttk.Scrollbar(table_wrapper, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        table_wrapper.columnconfigure(0, weight=1)
        table_wrapper.rowconfigure(0, weight=1)

        actions = ttk.Frame(self, padding=12)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="✍️ Rename Selected", command=self._rename_selected).pack(side=tk.LEFT)
        ttk.Button(actions, text="🚀 Rename All", command=self._rename_all).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="↩️ Rollback Last Rename", command=self._rollback).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.status_var, padding=(12, 0, 12, 12), foreground="#005a9c").pack(
            anchor="w"
        )

        self._handle_mode_change()

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
        self.status_var.set("Collecting files and requesting AI suggestions...")

        worker = threading.Thread(target=self._scan_worker, args=(folder,), daemon=True)
        worker.start()

    def _scan_worker(self, folder: Path) -> None:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]
        if not files:
            self.ui_queue.put(("status", "No image/video files found in selected folder."))
            return

        provider = AIProvider(
            mode=self.provider_mode_var.get(),
            endpoint=self.endpoint_var.get(),
            model=self.model_var.get(),
            api_key=self.api_key_var.get(),
        )

        date_text = format_date(self.date_format_var.get()) if self.include_date_var.get() else ""

        for idx, path in enumerate(files, start=1):
            try:
                media_bytes = extract_video_first_frame(path) if path.suffix.lower() in VIDEO_EXTENSIONS else load_image_bytes(path)
                suggestion = provider.suggest_name(media_bytes, path.stem)
                rec = FileSuggestion(
                    path=path,
                    original_name=path.name,
                    suggested_name=suggestion,
                    include_date=self.include_date_var.get(),
                    date_text=date_text,
                )
                self.ui_queue.put(("add", rec))
                self.ui_queue.put(("status", f"Suggested {idx}/{len(files)}: {path.name}"))
            except Exception as exc:  # noqa: BLE001
                self.ui_queue.put(("error_row", path.name, str(exc)))

        self.ui_queue.put(("status", f"Suggestion complete. {len(files)} files processed."))

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
            elif kind == "status":
                self.status_var.set(msg[1])

        self.after(80, self._process_ui_queue)

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
