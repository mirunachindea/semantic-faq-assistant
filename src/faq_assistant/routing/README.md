# routing

The semantic router: an ordered chain of rules, where the first rule to return a decision wins.

- `router.py` — `SemanticRouter`, `RoutingContext` and the default rule chain:
  1. `InputGuardRule` → compliance (deterministic, no model cost)
  2. `HighConfidenceMatchRule` → local (clear knowledge-base hit, skips the LLM)
  3. `LLMRouterRule` → local, llm or compliance (LLM judges intent, scope and safety)
  4. `ScoreFallbackRule` → local or llm (threshold backup if the LLM router abstains)

New policies are added by writing a `RoutingRule` subclass and inserting it into the chain.
