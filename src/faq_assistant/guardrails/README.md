# guardrails

Deterministic safety checks around the model calls. They complement, not replace, the LLM router's own safety classification and the hardened prompts.

- `input_guard.py` — `InputGuard` normalises user input and blocks known prompt-injection and abuse patterns without any model call.
- `output_guard.py` — `OutputGuard` rejects model outputs that are empty, oversized or leak the system prompt (detected via the canary token).
