# retrieval

Hybrid semantic search over the knowledge base.

- `search.py` — `HybridSearcher` embeds the query, retrieves dense candidates (cosine against both the question and a topic + question + answer document) and re-ranks them with a bounded lexical boost that can only raise a score, keeping results in `[dense, 1]`.
