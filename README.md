# Gemini CLI (API-key only)

## What this project is
- A simple, medium-level command-line app for Q&A using Google Gemini via REST API only.
- No Flask, no database, no browser/chat fallbacks. Strict API-key usage.

## Available models (API-only)
- Gemini 2.5 Pro: Advanced reasoning, best for coding, research, long context.
- Gemini 2.5 Flash: Fast, cost-efficient multimodal tasks.
- Gemini 2.5 Flash-Lite: Lightweight, cheaper, high-throughput.
- Gemini Nano: On-device only (NOT available via API key; rejected by this app).

The app asks you to choose a model for every question. If the model is unavailable over the API, it returns an error (no fallback to browser/chat or other providers).

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Provide your Gemini API key
export GOOGLE_API_KEY="AIza..."

# Run the CLI
python main.py
```

## Usage
- Type your question when prompted.
- Select a model for that question: 1) 2.5 Pro, 2) 2.5 Flash, 3) 2.5 Flash-Lite, or type a model id.
- If you select Gemini Nano, the app will error (Nano is on-device, not API-accessible).
- The app prints the answer and keeps a short in-memory history for the session.

Commands: `help`, `history`, `clear`, `exit`.

## Notes
- The app uses the REST endpoint `models/{model}:generateContent` and never falls back to browser/chat versions.
- If a model returns an error or no candidates, you will see an error message and can try a different model.

## Structure
- `main.py`: Gemini-only CLI logic.
- `requirements.txt`: minimal dependencies (requests, colorama, python-dotenv).
