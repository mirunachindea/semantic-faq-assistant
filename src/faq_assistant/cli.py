"""``faq-admin``: command-line management of the embeddings database.

Examples::

    faq-admin validate                             # curation report, no API calls
    faq-admin init-db                              # create extension + schema
    faq-admin sync                                 # embed only new/changed items
    faq-admin sync --collection billing --file billing.json --prune
    faq-admin sync --async                         # hand the job to a Celery worker
    faq-admin list
    faq-admin delete-collection billing --yes
    faq-admin search "how do I reset my password"
    faq-admin ask "how do I reset my password"
    faq-admin evaluate [--with-router]
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from faq_assistant.config import Settings, get_settings
from faq_assistant.container import create_container, create_indexer, create_store
from faq_assistant.domain.errors import FAQAssistantError
from faq_assistant.evaluation import (
    PrefetchedQueryEmbeddings,
    evaluate_retrieval,
    evaluate_routing,
    load_eval_cases,
)
from faq_assistant.knowledge_base.loader import curate_entries, load_knowledge_base
from faq_assistant.llm.factory import build_embeddings
from faq_assistant.logging_config import configure_logging


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


async def cmd_validate(settings: Settings, args: argparse.Namespace) -> None:
    """Show which entries would be indexed, fixed or rejected."""
    source = load_knowledge_base(args.file)
    report = curate_entries(source.knowledge_base_items, args.collection)
    _print(
        {
            "accepted": len(report.items),
            "rejected": [
                {"question": r.question, "issues": [asdict(i) for i in r.issues]}
                for r in report.rejected
            ],
            "warnings": {q: [asdict(i) for i in issues] for q, issues in report.warnings.items()},
        }
    )


async def cmd_init_db(settings: Settings, _args: argparse.Namespace) -> None:
    """Create the pgvector extension and schema."""
    store = await create_store(settings)
    await store.close()
    _print({"status": "ok", "vector_store": settings.vector_store})


async def cmd_sync(settings: Settings, args: argparse.Namespace) -> None:
    """Curate a knowledge-base file and embed new/changed items."""
    source = load_knowledge_base(args.file)
    if args.run_async:
        from faq_assistant.worker.tasks import sync_collection_task  # noqa: PLC0415

        raw = [item.model_dump() for item in source.knowledge_base_items]
        task = sync_collection_task.delay(args.collection, raw, prune=args.prune)
        _print({"task_id": task.id, "status": "queued"})
        return
    curated = curate_entries(source.knowledge_base_items, args.collection)
    store = await create_store(settings)
    try:
        indexer = create_indexer(settings, store, build_embeddings(settings))
        report = await indexer.sync(
            args.collection, curated.items, prune=args.prune, force=args.force
        )
    finally:
        await store.close()
    _print({**report.as_dict(), "rejected": [r.question for r in curated.rejected]})


async def cmd_list(settings: Settings, _args: argparse.Namespace) -> None:
    """List collections."""
    store = await create_store(settings)
    try:
        _print([asdict(c) for c in await store.list_collections()])
    finally:
        await store.close()


async def cmd_delete_collection(settings: Settings, args: argparse.Namespace) -> None:
    """Delete a collection and its items."""
    if not args.yes:
        msg = "Refusing to delete without --yes"
        raise SystemExit(msg)
    store = await create_store(settings)
    try:
        await store.delete_collection(args.name)
    finally:
        await store.close()
    _print({"deleted": args.name})


async def cmd_search(settings: Settings, args: argparse.Namespace) -> None:
    """Show retrieval candidates and scores for a query (no chat model calls)."""
    container = await create_container(settings)
    try:
        hits = await container.searcher.search(args.query, args.collection)
    finally:
        await container.aclose()
    _print(
        [
            {
                "question": h.item.question,
                "category": h.item.category,
                "score": h.score,
                "dense": h.dense_score,
                "lexical": h.lexical_score,
            }
            for h in hits
        ]
    )


async def cmd_ask(settings: Settings, args: argparse.Namespace) -> None:
    """Run a question through the full assistant."""
    container = await create_container(settings)
    try:
        answer = await container.assistant.ask(args.question)
    finally:
        await container.aclose()
    _print(answer.model_dump(mode="json"))


async def cmd_evaluate(settings: Settings, args: argparse.Namespace) -> None:
    """Evaluate retrieval (and optionally routing) on a labelled dataset."""
    cases = load_eval_cases(args.dataset)
    embeddings = PrefetchedQueryEmbeddings(build_embeddings(settings))
    container = await create_container(settings, embeddings=embeddings)
    try:
        await embeddings.prefetch([case.query for case in cases])
        retrieval = await evaluate_retrieval(container.searcher, settings.default_collection, cases)
        output: dict[str, Any] = {"retrieval": asdict(retrieval)}
        if args.with_router:
            output["routing"] = asdict(await evaluate_routing(container.assistant, cases))
    finally:
        await container.aclose()
    _print(output)


Command = Callable[[Settings, argparse.Namespace], Awaitable[None]]


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(prog="faq-admin", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Command, help_text: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        return command

    def add_source_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--file", type=Path, default=settings.knowledge_base_path)
        command.add_argument("--collection", default=settings.default_collection)

    add_source_args(add("validate", cmd_validate, "Curation report (no API calls)"))
    add("init-db", cmd_init_db, "Create extension and schema")

    sync = add("sync", cmd_sync, "Embed new/changed items (token-efficient upsert)")
    add_source_args(sync)
    sync.add_argument("--prune", action="store_true", help="Delete items missing from the file")
    sync.add_argument("--force", action="store_true", help="Re-embed everything")
    sync.add_argument("--async", dest="run_async", action="store_true", help="Queue on Celery")

    add("list", cmd_list, "List collections")
    delete = add("delete-collection", cmd_delete_collection, "Delete a collection")
    delete.add_argument("name")
    delete.add_argument("--yes", action="store_true", help="Confirm deletion")

    search = add("search", cmd_search, "Inspect retrieval scores for a query")
    search.add_argument("query")
    search.add_argument("--collection", default=settings.default_collection)

    ask = add("ask", cmd_ask, "Ask a question through the full pipeline")
    ask.add_argument("question")

    evaluate = add("evaluate", cmd_evaluate, "Evaluate retrieval/routing on a labelled set")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval/eval_set.jsonl"))
    evaluate.add_argument(
        "--with-router", action="store_true", help="Also evaluate routing (uses the chat model)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings).parse_args(argv)
    try:
        asyncio.run(args.handler(settings, args))
    except FAQAssistantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
