# llm

Everything that talks to language and embedding models, kept provider-agnostic through LangChain interfaces.

- `factory.py` — builds the chat and embedding models from configuration (`LLM_PROVIDER`, `CHAT_MODEL`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`).
- `prompts.py` — prompt templates, hardened against prompt injection (untrusted input delimited in the human turn, instruction hierarchy, per-process canary token).
- `chains.py` — composes prompts, model and parsers into the typed runnables the app uses (`LLMChains`, `RouterVerdict`).
- `errors.py` — wraps model calls and translates provider exceptions into domain errors (including rate limiting).
