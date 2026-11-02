import argparse
import json
from pathlib import Path
from .data import load_requests

from .core import (BGEEncoder, EncoderConfig, DenseRetriever, Baseline,
                   PreviewGenerator, HTTPGenerator, read_jsonl)


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
        enc = BGEEncoder(EncoderConfig(model=args.model, revision=args.revision), args.device)
        result = DenseRetriever.build(args.chunks, args.index, enc, args.split, args.upstream_manifest)
        print(json.dumps(result.manifest, indent=2))
        return
    retriever = DenseRetriever.load(args.index, device=args.device)
    if args.command == "retrieve":
        print(json.dumps(retriever.retrieve(args.question, args.top_k, args.paper_id), ensure_ascii=False, indent=2))
        return
    if args.generator == "http" and (not args.endpoint or not args.llm_model):
        parser.error("http requires --endpoint and --llm-model")
    generator = HTTPGenerator(args.endpoint, args.llm_model) if args.generator == "http" else PreviewGenerator()
    baseline = Baseline(retriever, generator, args.top_k, args.context_chars, args.scope, task=args.task)
    requests = load_requests(args.questions, args.split)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids destroying prior experiment results.
    with open(args.output, "x", encoding="utf-8") as f:
        for request in requests:
            result = baseline.run(request)  # Fail fast; completed rows remain on disk.
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
