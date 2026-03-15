# AI File Namer (Windows GUI)

A Tkinter desktop app to bulk-rename images and videos using an AI model.

## Features
- Select a folder and scan image/video files.
- Configure AI as:
  - **Local**: Ollama `/api/generate` endpoint (example: `llava`).
  - **Remote**: OpenAI-compatible `/v1/chat/completions` endpoint.
- For videos, extracts the **first frame** and sends it to AI.
- Preview original filename, AI suggestion, and final filename.
- Include optional date prefix using a custom `strftime` format.
- Rename selected or all files with confirmation.
- Roll back the last rename batch.

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/ai_file_namer.py
```

## Notes
- The model should return concise names. The app sanitizes output for Windows-safe filenames.
- Keep local/remote AI endpoint available for fast response.
