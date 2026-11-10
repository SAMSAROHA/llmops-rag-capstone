"""Command line entry point: ingest, ask, eval, gate, prompts, costs, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import PROJECT_ROOT, get_settings


def _pipeline(settings, with_ledger=True):
    from .pipeline import RAGPipeline
    from .tracing import setup_tracing

    setup_tracing(settings)
    return RAGPipeline.from_settings(settings, with_ledger=with_ledger)


def cmd_ingest(args, settings) -> int:
    from .embeddings import build_embedder
    from .ingest import ingest
    from .vectorstore import VectorStore, build_client

    store = VectorStore(build_client(settings), build_embedder(settings), settings.collection)
    n = ingest(
        store,
        Path(args.path or settings.docs_dir),
        settings.chunk_chars,
        settings.chunk_overlap,
        recreate=args.recreate,
    )
    print(f"Indexed {n} chunks into '{settings.collection}' ({store.count()} total).")
    return 0


def cmd_ask(args, settings) -> int:
    result = _pipeline(settings).answer(args.question, args.top_k, args.prompt_version)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(result.answer)
    print("\nSources:")
    for i, c in enumerate(result.sources, start=1):
        print(f"  [{i}] {c.source} (score {c.score:.3f})")
    u = result.usage
    print(
        f"\nprompt {result.prompt.name}/{result.prompt.version}  model {result.model}  "
        f"tokens {u.input_tokens} in / {u.output_tokens} out  "
        f"cost ${result.cost.total_usd:.6f}  {result.latency_ms:.0f} ms  "
        f"trace {result.trace_id}"
    )
    return 0


def _evaluate(args, settings, gated: bool):
    from .evaluation import build_judge, load_gate, load_golden, run_evaluation

    # Eval traffic stays out of the production cost ledger.
    pipeline = _pipeline(settings, with_ledger=False)
    if pipeline.store.count() == 0:
        sys.exit("The vector store is empty. Run `ragapp ingest` first.")
    report = run_evaluation(
        pipeline,
        build_judge(settings),
        load_golden(args.golden or settings.golden_path),
        prompt_version=args.prompt_version,
        gate_config=load_gate(args.gate_file or settings.gate_file) if gated else None,
    )
    out = report.save(args.out)
    markdown = report.to_markdown()
    print(markdown)
    for name, value in report.metrics.items():
        print(f"{name:28s} {'n/a' if value is None else f'{value:.4f}'}")
    print(f"\nReport: {out}   MLflow run: {report.mlflow_run_id}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown)
    return report


def cmd_eval(args, settings) -> int:
    _evaluate(args, settings, gated=False)
    return 0


def cmd_gate(args, settings) -> int:
    report = _evaluate(args, settings, gated=True)
    if report.gate.passed:
        print("\nGATE PASSED: release may proceed.")
        return 0
    failed = ", ".join(c.metric for c in report.gate.checks if not c.passed)
    print(f"\nGATE FAILED ({failed}): release blocked.", file=sys.stderr)
    return 1


def cmd_prompts(args, settings) -> int:
    from .prompts import PromptRegistry

    registry = PromptRegistry(settings.prompts_dir)
    if args.action == "list":
        active = registry.get(settings.prompt_name, settings.prompt_version).version
        for version in registry.versions(settings.prompt_name):
            p = registry.get(settings.prompt_name, version)
            marker = "*" if version == active else " "
            print(f"{marker} {p.name}/{p.version}  sha {p.sha}  {p.description}")
    else:
        from .tracing import setup_tracing

        setup_tracing(settings)
        registered = registry.sync_to_mlflow(settings.prompt_name)
        print(f"Registered in MLflow: {registered or 'nothing new'}")
    return 0


def cmd_costs(args, settings) -> int:
    from .cost import CostLedger

    print(json.dumps(CostLedger(settings.cost_db_path).summary(), indent=2))
    return 0


def cmd_serve(args, settings) -> int:
    import uvicorn

    uvicorn.run("ragapp.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragapp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk and index documents into Qdrant")
    p.add_argument("path", nargs="?", help="file or folder of .md/.txt/.pdf (default: data/docs)")
    p.add_argument("--recreate", action="store_true", help="drop the collection first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--top-k", type=int)
    p.add_argument("--prompt-version")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    for name, func, text in (
        ("eval", cmd_eval, "run the golden set and log scores to MLflow"),
        ("gate", cmd_gate, "run the golden set and exit 1 if thresholds are not met"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--prompt-version", help="prompt version to evaluate (default: active)")
        p.add_argument("--golden", help="path to golden .jsonl")
        p.add_argument("--gate-file", help="path to gate thresholds .yaml")
        p.add_argument("--out", default=str(PROJECT_ROOT / "reports"))
        p.set_defaults(func=func)

    p = sub.add_parser("prompts", help="list prompt versions or sync them to MLflow")
    p.add_argument("action", choices=["list", "sync"])
    p.set_defaults(func=cmd_prompts)

    p = sub.add_parser("costs", help="print the cost ledger summary")
    p.set_defaults(func=cmd_costs)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args, get_settings())


if __name__ == "__main__":
    sys.exit(main())
