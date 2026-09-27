# storage

Vector store abstraction and its implementations. The rest of the app depends only on the protocol, so backends are swappable.

- `base.py` — the `VectorStore` protocol plus `EmbeddedItem` and `CollectionInfo`.
- `memory.py` — `InMemoryVectorStore`, a brute-force numpy store for tests and dependency-free local runs.
- `postgres.py` — `PostgresVectorStore`, backed by PostgreSQL + pgvector with HNSW cosine indexes.
