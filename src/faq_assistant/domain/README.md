# domain

Core types shared by every layer. No dependencies on frameworks or providers.

- `models.py` — `FAQItem`, `SearchHit`, `Route`, `RouteDecision`, `AnswerSource` and `AssistantAnswer`.
- `errors.py` — the application exception hierarchy (`FAQAssistantError` and subclasses for upstream, vector-store and knowledge-base failures), mapped to HTTP responses by the API layer.
