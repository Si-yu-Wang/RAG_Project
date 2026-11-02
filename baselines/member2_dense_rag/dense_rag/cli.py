import argparse
import json
import sys
from pathlib import Path
from .data import load_requests

from .core import (BGEEncoder, EncoderConfig, DenseRetriever, Baseline,
                   PreviewGenerator, HTTPGenerator)


def main():
    parser = argparse.ArgumentParser(description="Member 2 dense RAG baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--chunks", required=True)
    build.add_argument("--index", required=True)
    build.add_argument("--model", default=EncoderConfig.model)
    build.add_argument("--revision", default=EncoderConfig.revision)
    build.add_argument("--device", default="cpu")
    build.add_argument("--split")
    build.add_argument("--upstream-manifest")
    for name in ("retrieve", "run"):
        p = sub.add_parser(name)
        p.add_argument("--index", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--top-k", type=int, default=5)
        if name == "retrieve":
            p.add_argument("--question", required=True)
            p.add_argument("--paper-id")
        else:
            p.add_argument("--questions", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--scope", choices=["paper", "global"], default="paper")
            p.add_argument("--context-chars", type=int, default=12000)
            p.add_argument("--generator", choices=["preview", "http"], default="preview")
            p.add_argument("--endpoint")
            p.add_argument("--llm-model")
            p.add_argument("--split")
            p.add_argument("--task", choices=["qa", "pubmedqa"], default="qa")
    args = parser.parse_args()
    if args.command == "build":
        if Path(args.index).exists():
            parser.error("Index directory already exists; choose a new --index path")
        enc = BGEEncoder(EncoderConfig(model=args.model, revision=args.revision), args.device)
        result = DenseRetriever.build(args.chunks, args.index, enc, args.split, args.upstream_manifest)
        print(json.dumps(result.manifest, indent=2))
        return
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.command == "run":
        if args.context_chars < 1:
            parser.error("--context-chars must be positive")
        if args.generator == "http" and (not args.endpoint or not args.llm_model):
            parser.error("http requires --endpoint and --llm-model")
        if Path(args.output).exists():
            parser.error("Output already exists; choose a new --output path")
        requests = load_requests(args.questions, args.split)
    retriever = DenseRetriever.load(args.index, device=args.device)
    if args.command == "retrieve":
        print(json.dumps(retriever.retrieve(args.question, args.top_k, args.paper_id), ensure_ascii=False, indent=2))
        return
    generator = HTTPGenerator(args.endpoint, args.llm_model) if args.generator == "http" else PreviewGenerator()
    baseline = Baseline(retriever, generator, args.top_k, args.context_chars, args.scope, task=args.task)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids destroying prior experiment results.
    with open(args.output, "x", encoding="utf-8") as f:
        for number, request in enumerate(requests, 1):
            result = baseline.run(request)  # Fail fast; completed rows remain on disk.
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            if number == 1 or number % 10 == 0 or number == len(requests):
                print(f"Completed {number}/{len(requests)} questions", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
