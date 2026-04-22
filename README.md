# llm-redteaming

LangGraph agent demo (`main.py`) + PyRIT harness (`pyrit_test.py`).

![langsmith tracing](assets/pyritreport.png)

- **Target** — your app (`POST /chat`). That is what you are testing.
- **Attacker** — PyRIT when you run `pyrit_test.py` (red-team LLM + probes against `/chat`). **Judge** — the scorer inside PyRIT that labels each target reply (same OpenAI account as in `.env`).

![lPyrit report](assets/langsmith.png)
## Run

```bash
uv sync
cp .env.sample .env   # fill in keys
uv run uvicorn main:app --host 127.0.0.1 --port 8000
uv run python pyrit_test.py   # second terminal; hits /chat
```
To see langsmith tracing (optional)
```bash
langgraph dev
```

## Env

Use **`.env.sample`** as the template (LangSmith + `OPENAI_API_KEY` for the app).

For **PyRIT**: if you only set `OPENAI_API_KEY`, `pyrit_test.py` copies it to what PyRIT expects. Optional: `EXPENSE_API_URL` (default `http://127.0.0.1:8000`), `REDTEAM_MAX_TURNS` (default `8`).

## CI output

GitHub Actions runs `.github/workflows/pyrit-redteam.yml`; view the PyRIT report directly in the job logs, and download `uvicorn-log` from run artifacts if startup/runtime fails.



