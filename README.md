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

For **PyRIT**: if you only set `OPENAI_API_KEY`, `pyrit_test.py` copies it to what PyRIT expects. Optional: `EXPENSE_API_URL` (default `http://127.0.0.1:8000`), `REDTEAM_MAX_TURNS` (default `5`).

## ASR + GitHub Action 

- **ASR (Attack Success Rate)** = percent of attacks where the objective scorer says `SUCCESS`.
- `pyrit_test.py` prints ASR in the session analytics line:
  - `Overall: success_rate=...%`
- CI workflow `.github/workflows/pyrit-redteam.yml` parses that ASR and compares it to `PYRIT_ASR_FAIL_THRESHOLD` (default `0.35`).
- If ASR is higher than the threshold, the workflow fails and the PR can be blocked from merge (when this check is required in branch protection).

## CI output

GitHub Actions runs `.github/workflows/pyrit-redteam.yml`; view the PyRIT report directly in the job logs, and download `uvicorn-log` from run artifacts if startup/runtime fails.

# Notes: This is a very simple agent to experiment redteaming with PyRIT. It is not a production ready agent.

