# routers

FastAPI routers, one per endpoint group.

- `questions.py` — `POST /ask-question`: answers from the knowledge base, falls back to the LLM, or refuses.
- `admin.py` — knowledge-base management (admin token required): list collections, queue ingestion jobs, check job status.
- `health.py` — liveness and readiness probes.
