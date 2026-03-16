# AI File Namer (Windows GUI)

A Tkinter desktop app to bulk-rename images and videos using an AI model.

## Features
- Select a folder and scan image/video files (optionally recursive through subfolders).
- Configure AI as:
  - **Local**: Ollama `/api/generate` endpoint (example: `llava`).
  - **Remote**: OpenAI-compatible `/v1/chat/completions` endpoint using OpenAI OAuth login (no API key required).
- Built-in **OpenAI OAuth (Authorization Code + PKCE)** button opens your browser, captures the callback locally, and stores the returned access token in app settings for future sessions.
- For videos, extracts the **first frame** and sends it to AI.
- Preview original filename, AI suggestion, and final filename with clickable column sorting.
- Include optional date prefix using a custom `strftime` format.
- Rename selected or all files with confirmation.
- Roll back the last rename batch.
- Suggest AI names for folders based on each directory's full contents and apply recursive folder categorisation.
- Suggest and apply AI-driven whole-tree restructure plans that consolidate messy folders, move files into logical categories, and clean duplicate files.

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


## Download installers
Get the latest generated installers for both platforms from GitHub Releases:
- **Latest release:** [releases/tag/latest](releases/tag/latest)
- **All releases:** [releases](releases)


## Notes
- The model should return concise names. The app sanitizes output for Windows-safe filenames.
- For local mode, install Ollama: https://ollama.com/download
- Remote mode requires an OpenAI account: https://platform.openai.com/signup
- Keep local/remote AI endpoint available for fast response.

## FAQ: "[truncated ... chars]" in AI debug logs
- Most systems label long payloads as `...[truncated N chars]` **only in logs/UI output** to keep logs readable and avoid huge terminal/UI rendering costs.
- In those cases, the model usually still receives the full request/response body that was sent over the API.
- The only way to be sure is to check transport-level facts:
  - request/response size limits in your provider docs,
  - HTTP status codes/errors (for example 400/413 for payload too large),
  - token-limit errors returned by the model endpoint.
- If you are near limits, chunking is a valid approach:
  - split large context into ordered chunks,
  - include a stable item ID and chunk index (e.g., `doc-42 chunk 3/8`),
  - ask the model to acknowledge each chunk,
  - then send a final synthesis request that references the acknowledged chunks.
- Chunking works best when each chunk is self-contained and you repeat critical instructions (goal, output format) in each call.

## CI: Build installers automatically
A GitHub Actions workflow is included at `.github/workflows/build-installers.yml`.

It runs on pushes, PRs, and manual dispatch and produces:
- **Windows**
  - `AIFileNamer.exe` via PyInstaller (`--onefile`, self-contained)
  - `.msi` installer via cx_Freeze (`bdist_msi`)
- **macOS**
  - `AIFileNamer.app` via PyInstaller
  - `AIFileNamer-macOS.pkg` installer via `pkgbuild`

All outputs are uploaded as workflow artifacts:
- `windows-installers`
- `macos-installers`

> Note: unsigned macOS/Windows installers may trigger security prompts until code-signing is configured.
