# Chatbot Multi-Model (Intermediate)

## What this project is
- A Flask web app for Q&A and image uploads
- Supports two AI providers via env: Groq or Gemini
- Stores interactions in SQLite
- Includes a CLI client (`main.py`)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Choose ONE provider
# Groq
export AI_PROVIDER=groq
export GROQ_API_KEY="gsk_..."

# OR Gemini
export AI_PROVIDER=gemini
export GOOGLE_API_KEY="AIza..."
export GEMINI_MODEL="gemini-1.5-flash"

export PORT=5001
python app.py
```

Open `http://127.0.0.1:5001`.

## Structure
- `app.py`: Flask app (routes, DB init, provider calls)
- `main.py`: CLI client for interactive Q&A
- `templates/*`: HTML pages
- `static/uploads`: saved images
- `instance/database.db`: main SQLite database

## Key routes
- `/`: Ask a question
- `/ask`: Handles Q&A, saves to DB
- `/history`: View past Q&A
- `/upload_image`: Upload image
- `/ask_about_image`: Ask about uploaded image
- `/image_history`: See image history
- `/health`: Basic health JSON
- `/config_debug`: Shows provider and config presence flags

## Interview talking points
- Provider abstraction: `AI_PROVIDER` switches between Groq (OpenAI-compatible chat completions) and Gemini (Google Generative Language API). We pass messages/parts to each API’s REST endpoint. Models differ: Groq uses `llama-*`, Gemini uses `gemini-*`.
- Data model: SQLite tables for `interactions`, `images`, `image_qa`. Web app uses `instance/database.db`. Some legacy code references `interactions.db`; migration to a single DB is straightforward.
- Image handling: Uses Pillow/NumPy. We extract simple features (dimensions, size, color stats). We do not send raw image bytes to the LLM; we craft a descriptive prompt from technical metadata.
- Error handling: Warnings for missing keys; HTTP status checks; fallback messages. Health/config endpoints aid diagnostics without exposing secrets.
- Security: Never log API keys. File uploads use `secure_filename`. Max upload size configured.
- Extensibility: Provider module can be introduced to fully abstract AI calls (e.g., add OpenAI, Anthropic). Templates can be extended for chat history and streaming.

## Common pitfalls
- Using a Gemini key (`AIza...`) with Groq or vice versa. Ensure the right provider+key.
- Port 5000 busy: set `PORT=5001`.
- “command not found: python”: activate venv or use `python3`.

## CLI usage
```bash
source .venv/bin/activate
python main.py
```

Commands: `help`, `list_models`, `switch_model`, `code_mode`, `history`, `analyze`, `view`, `save`, `load`.
