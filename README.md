# MedScope-
MedScope — Multi-agent AI that explains diseases and surfaces latest research in plain, patient-friendly language.

## Run With Docker Compose

1. Ensure Docker Desktop is running.
2. Add required keys to `.env` (for example `GROQ_API_KEY`; optional `GEMINI_API_KEY` for Gemini embeddings).
3. Build and start:

```bash
docker compose up --build
```

4. Open the app at http://localhost:8501

To stop:

```bash
docker compose down
```

## Run Without Any LLM API Key (Docker + Ollama)

This mode runs the language model fully inside Docker containers and does not require GROQ or OpenAI keys.

1. Start Ollama container:

```bash
docker compose up -d ollama
```

2. Pull the model into the Ollama container (first run only):

```bash
docker compose exec ollama ollama pull mistral
```

3. Start the app container:

```bash
docker compose up -d medscope
```

4. Open the app at http://localhost:8501

Notes:
- The project is configured to call Ollama at http://ollama:11434 from inside Docker.
- You can change model by setting OLLAMA_MODEL in docker-compose.yml.
