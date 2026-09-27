# api

FastAPI layer: HTTP contract, authentication and error mapping. Endpoints live in [`routers/`](routers/).

- `schemas.py` — request/response models of the public API (`AskQuestionRequest`, `AskQuestionResponse`, ingestion and health payloads).
- `dependencies.py` — FastAPI dependencies: access to the container, settings and assistant service, plus bearer-token and admin-token checks.
- `errors.py` — maps domain exceptions to HTTP responses; error bodies never expose internal details and are correlated through `X-Request-ID`.
