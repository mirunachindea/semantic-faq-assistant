# agents

Responders that produce the final answer once the router has picked a route.

- `responders.py` — the `Responder` interface and its three implementations:
  - `LocalAnswerAgent` answers from the matched knowledge-base entry, optionally personalised by the LLM.
  - `LLMAnswerAgent` answers in-scope questions that the knowledge base does not cover.
  - `ComplianceAgent` returns the fixed refusal message for out-of-scope or unsafe questions.
