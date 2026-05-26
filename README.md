# MedScope-
MedScope — Multi-agent AI that explains diseases and surfaces latest research in plain, patient-friendly language.

## Run With Docker Compose (Fully Local, No API Keys)

This setup runs the LLM and embeddings locally in Docker using Ollama.
Models are pulled automatically on startup.

1. Ensure Docker Desktop is running.
2. Build and start:

```bash
docker compose up --build
```

3. Open the app at http://localhost:8501

To stop:

```bash
docker compose down
```

Notes:
- The project calls Ollama at http://ollama:11434 from inside Docker.
- The helper service automatically pulls `mistral` and `nomic-embed-text`.
- You can change models by setting `OLLAMA_MODEL` and `OLLAMA_EMBED_MODEL` in docker-compose.yml.
