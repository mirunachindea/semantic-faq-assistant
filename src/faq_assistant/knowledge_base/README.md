# knowledge_base

Loading, curating and embedding the FAQ source data.

- `loader.py` — reads and validates the knowledge-base JSON, then curates entries (clean, validate, de-duplicate) into `FAQItem`s with stable ids and a `CurationReport` of rejected entries.
- `cleaning.py` — text normalisation and small, pure data-quality rules (placeholder questions, non-informative answers, emoji noise, unsafe example passwords).
- `indexer.py` — `EmbeddingIndexer` keeps a vector-store collection in sync incrementally: only new or changed items (by content hash) are embedded; pruning stale items is opt-in.
