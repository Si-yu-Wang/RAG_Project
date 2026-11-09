#!/usr/bin/env bash
# PubMedQA (citation-graph corpus) only: encode -> build ReaRev graph -> train -> eval -> Dense/BM25 -> score.
# Same GNN-RAG source, Python environment and BGE cache as the QASPER run.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GNN_DIR="${GNN_DIR:-$(dirname -- "$PROJECT")/GNN-RAG/gnn}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
STAGE="${1:?Usage: run_pubmedqa.sh export|embed|build|smoke|train|eval|dense|score|tune|all}"
RUN_ID="${RUN_ID:-pubmedqa_rearev_s1}"
[[ "$STAGE" != smoke ]] || RUN_ID="${RUN_ID}_smoke"
[[ "$RUN_ID" =~ ^pubmedqa_[A-Za-z0-9_-]+$ ]] || { echo 'RUN_ID must start with pubmedqa_'; exit 1; }
RUN_DIR="$PROJECT/runs/$RUN_ID"
PREP="$PROJECT/runs/pubmedqa_prepare"
GRAPH="$PROJECT/data/processed/pubmedqa_gnn"
TOOL="$PROJECT/data_process/pubmedqa_to_gnn.py"
DENSE_RANK="$PROJECT/rankings/pubmedqa_test_dense_bge.jsonl"
BM25_RANK="$PROJECT/rankings/pubmedqa_test_bm25.jsonl"
SEED="${SEED:-19960626}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
# ReaRev hyperparameters (env-overridable). `train` saves them to runs/<RUN_ID>/hparams.env and
# every later stage on that run reloads them, so eval always rebuilds the trained architecture.
ENTITY_DIM="${ENTITY_DIM:-128}"
NUM_ITER="${NUM_ITER:-2}"
NUM_INS="${NUM_INS:-3}"
NUM_GNN="${NUM_GNN:-2}"
LR="${LR:-0.0005}"
# Which dev-selected ReaRev checkpoint to evaluate: f1 (default; rewards ranking ALL gold chunks of the
# source paper high) or h1 (top-1 only: near ceiling on this task and noisy on 92 dev questions).
CKPT_METRIC="${CKPT_METRIC:-f1}"
if [[ "$STAGE" == eval && -f "$RUN_DIR/hparams.env" ]]; then
    # shellcheck disable=SC1091
    source "$RUN_DIR/hparams.env"
    echo "Loaded $RUN_ID hparams: entity_dim=$ENTITY_DIM num_iter=$NUM_ITER num_ins=$NUM_INS num_gnn=$NUM_GNN lr=$LR seed=$SEED ckpt=$CKPT_METRIC"
fi
TUNE_EPOCHS="${TUNE_EPOCHS:-60}"
# num_gnn:num_iter:entity_dim. Same-paper chunks form a chain (~3.4 chunks) and source papers meet
# only through shared references (2 hops), so deeper propagation is tested; 2:1:256 = QASPER's best.
TUNE_GRID="${TUNE_GRID:-3:2:50 2:2:128 2:2:256 3:2:128 3:2:256 2:1:256}"
mkdir -p "$PREP" "$PROJECT/rankings" "$PROJECT/reports"

fresh() { for p in "$@"; do [[ ! -e "$p" ]] || { echo "Output exists; preserve it before retrying: $p"; exit 1; }; done; }
cuda_check() { python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones((32,32),device="cuda"); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==32; print(torch.__version__, torch.cuda.get_device_name(0), "CUDA compute OK")'; }
logged() {
    local prefix="$1"; shift
    fresh "$prefix.command.txt" "$prefix.console.log"
    printf '%q ' "$@" > "$prefix.command.txt"; printf '\n' >> "$prefix.command.txt"
    date -Is > "$prefix.started.txt"
    "$@" 2>&1 | tee "$prefix.console.log"
    date -Is > "$prefix.completed.txt"
}
check_patch() {
    python - "$GNN_DIR" <<'PY'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
r = json.loads((d/'webqsp_patch_record.json').read_text())
assert r['upstream_commit'] == '28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4'
for name, hashes in r['files'].items():
    assert hashlib.sha256((d/name).read_text(encoding='utf-8').encode()).hexdigest() == hashes['patched_sha256_lf'], name
print('GNN-RAG source patch verified (training evaluates dev only).')
PY
}
graph_ready() {
    for f in train.json dev.json test.json entities.txt relations.txt node_feat.npy id_map.parquet vocab.txt word_emb.npy splits.json manifest.json; do
        test -s "$GRAPH/$f" || { echo "Missing $GRAPH/$f; run the build stage first"; exit 1; }
    done
}
gnn_args=(ReaRev --name pubmedqa --data_folder "$GRAPH/" --lm sbert --lm_frozen 1
    --relation_word_emb False --entity_emb_file node_feat.npy --kge_frozen 1
    --entity_dim "$ENTITY_DIM" --num_iter "$NUM_ITER" --num_ins "$NUM_INS" --num_gnn "$NUM_GNN"
    --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
    --eval_every 2 --lr "$LR" --seed "$SEED" --checkpoint_dir "$RUN_DIR")

case "$STAGE" in
    export)
        fresh "$GRAPH/_embed/chunks.jsonl" "$GRAPH/_embed/queries.jsonl"
        python "$TOOL" export --out "$GRAPH"
        ;;
    embed)
        cuda_check
        extra=(); [[ "${ALLOW_DOWNLOAD:-0}" != 1 ]] || extra=(--allow-download)
        for kind in chunks queries; do
            if [[ -s "$GRAPH/_embed/$kind.npy" ]]; then echo "skip (done): $kind embeddings"; continue; fi
            q=(); [[ "$kind" != queries ]] || q=(--query)
            logged "$PREP/embed-$kind" python "$PROJECT/data_process/embed_bge.py" --input "$GRAPH/_embed/$kind.jsonl" \
                --out "$GRAPH/_embed/$kind" --device cuda --batch-size 32 "${q[@]}" "${extra[@]}"
        done
        ;;
    build)
        test -s "$GRAPH/_embed/chunks.npy" && test -s "$GRAPH/_embed/queries.npy"
        logged "$PREP/build" python "$TOOL" build --out "$GRAPH"
        ;;
    smoke|train)
        graph_ready
        cuda_check
        check_patch
        fresh "$RUN_DIR"
        mkdir -p "$RUN_DIR"
        printf 'ENTITY_DIM=%q\nNUM_ITER=%q\nNUM_INS=%q\nNUM_GNN=%q\nLR=%q\nSEED=%q\nBATCH_SIZE=%q\nTEST_BATCH_SIZE=%q\nCKPT_METRIC=%q\n' \
            "$ENTITY_DIM" "$NUM_ITER" "$NUM_INS" "$NUM_GNN" "$LR" "$SEED" "$BATCH_SIZE" "$TEST_BATCH_SIZE" "$CKPT_METRIC" > "$RUN_DIR/hparams.env"
        cp "$GRAPH/manifest.json" "$RUN_DIR/graph-manifest.json"
        python -m pip freeze > "$RUN_DIR/environment.txt"
        args=("${gnn_args[@]}" --experiment_name "$RUN_ID")
        if [[ "$STAGE" == smoke ]]; then
            (cd "$GNN_DIR"; logged "$RUN_DIR/$RUN_ID" python -u main.py "${args[@]}" --num_epoch 2 --max_train 64)
            test -s "$RUN_DIR/$RUN_ID-final.ckpt"
            echo "Smoke passed: ReaRev loads the PubMedQA graph, trains 2 epochs and saves a checkpoint."
        else
            (cd "$GNN_DIR"; logged "$RUN_DIR/$RUN_ID" python -u main.py "${args[@]}" --num_epoch "$EPOCHS")
            test -s "$RUN_DIR/$RUN_ID-$CKPT_METRIC.ckpt"
            sha256sum "$RUN_DIR/$RUN_ID-$CKPT_METRIC.ckpt" > "$RUN_DIR/selected_checkpoint.sha256"
            echo "Training complete; dev-$CKPT_METRIC checkpoint selected. Next: RUN_ID=$RUN_ID run_pubmedqa.sh eval"
        fi
        ;;
    eval)
        graph_ready
        cuda_check
        check_patch
        test -s "$RUN_DIR/$RUN_ID-$CKPT_METRIC.ckpt"
        test -s "$DENSE_RANK" || bash "$0" dense
        fresh "$PROJECT/rankings/${RUN_ID}_test.jsonl"
        (cd "$GNN_DIR"; logged "$RUN_DIR/${RUN_ID}_eval" python -u main.py "${gnn_args[@]}" \
            --experiment_name "${RUN_ID}_eval" --is_eval --load_experiment "$RUN_ID-$CKPT_METRIC.ckpt" --eps 2)
        python "$TOOL" from-gnn --gnn-dir "$GRAPH" --info "$RUN_DIR/${RUN_ID}_eval_test.info" --dense "$DENSE_RANK" \
            --method "$RUN_ID" --out "$PROJECT/rankings/${RUN_ID}_test.jsonl"
        ;;
    dense)
        graph_ready
        [[ -s "$DENSE_RANK" ]] || python "$TOOL" dense --out-dir "$GRAPH" --method dense_bge --out "$DENSE_RANK"
        [[ -s "$BM25_RANK" ]] || python "$TOOL" bm25 --out-dir "$GRAPH" --out "$BM25_RANK"
        python "$TOOL" score --out-dir "$GRAPH" "$DENSE_RANK" "$BM25_RANK"
        ;;
    score)
        GNN_RANK="$PROJECT/rankings/${RUN_ID}_test.jsonl"
        test -s "$GNN_RANK" && test -s "$DENSE_RANK" && test -s "$BM25_RANK"
        python "$TOOL" score --out-dir "$GRAPH" "$DENSE_RANK" "$GNN_RANK" "$BM25_RANK" --out "$PROJECT/reports/${RUN_ID}_vs_dense.json"
        python "$TOOL" compare --out-dir "$GRAPH" "$DENSE_RANK" "$GNN_RANK" --out "$PROJECT/reports/${RUN_ID}_vs_dense_ci.json"
        ;;
    all)
        for s in export embed build smoke; do
            case "$s" in
                export) [[ -s "$GRAPH/_embed/queries.jsonl" ]] && { echo "skip (done): export"; continue; } ;;
                build) [[ -s "$GRAPH/manifest.json" ]] && { echo "skip (done): build"; continue; } ;;
                smoke) [[ -s "${RUN_DIR}_smoke/${RUN_ID}_smoke-final.ckpt" ]] && { echo "skip (done): smoke"; continue; } ;;
            esac
            bash "$0" "$s"
        done
        bash "$0" dense
        [[ -s "$RUN_DIR/selected_checkpoint.sha256" ]] || bash "$0" train
        [[ -s "$PROJECT/rankings/${RUN_ID}_test.jsonl" ]] || bash "$0" eval
        bash "$0" score
        ;;
    tune)
        graph_ready
        for spec in $TUNE_GRID; do
            IFS=: read -r g i d <<< "$spec"
            run="pubmedqa_tune_g${g}_i${i}_d${d}"
            if [[ -s "$PROJECT/runs/$run/selected_checkpoint.sha256" ]]; then echo "skip (trained): $run"; continue; fi
            if [[ -e "$PROJECT/runs/$run" ]]; then echo "SKIP incomplete/failed run (move it aside to retry): runs/$run"; continue; fi
            echo "=== train $run (num_gnn=$g num_iter=$i entity_dim=$d, $TUNE_EPOCHS epochs) ==="
            RUN_ID="$run" EPOCHS="$TUNE_EPOCHS" NUM_GNN="$g" NUM_ITER="$i" ENTITY_DIM="$d" CKPT_METRIC=f1 bash "$0" train \
                || echo "FAILED: $run (continuing with the rest of the grid)"
        done
        BEST="$(python - "$PROJECT/runs" "$PROJECT/reports/pubmedqa_tune_summary.json" $TUNE_GRID <<'PY'
import json, pathlib, re, sys
runs, out, specs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
pat = re.compile(r"EVAL F1: ([0-9.]+), H1: ([0-9.]+)")
rows = []
for spec in specs:
    g, i, d = spec.split(":")
    run = f"pubmedqa_tune_g{g}_i{i}_d{d}"
    folder = runs / run
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.glob("*.log"))] if folder.exists() else []
    evals = max(([(float(a), float(b)) for a, b in pat.findall(t)] for t in texts), key=len, default=[])
    row = {"run": run, "num_gnn": int(g), "num_iter": int(i), "entity_dim": int(d),
           "trained": (folder / "selected_checkpoint.sha256").exists(), "evals": len(evals)}
    if evals:
        k = max(range(len(evals)), key=lambda j: (evals[j][0], -j))  # first best F1 = ReaRev's -f1.ckpt epoch
        last = evals[-5:]
        row.update(best_dev_f1=evals[k][0], h1_at_best_f1=evals[k][1], best_epoch=2 * (k + 1),
                   best_dev_h1=max(e[1] for e in evals),
                   mean_last5_dev_f1=sum(e[0] for e in last) / len(last),
                   mean_last5_dev_h1=sum(e[1] for e in last) / len(last))
    rows.append(row)
ok = [r for r in rows if r["trained"] and "best_dev_f1" in r]
if not ok:
    sys.exit("No completed tuning run")
best = max(ok, key=lambda r: (r["best_dev_f1"], -r["entity_dim"], -r["num_gnn"], -r["num_iter"]))
out.write_text(json.dumps({"selection": "max dev F1 (ReaRev native; dev split of the fixed 60/10/30 question split); "
                                        "evaluated with that run's -f1.ckpt; test not used",
                           "selected": best["run"], "runs": rows}, indent=2), encoding="utf-8")
print("| run | gnn | iter | dim | best dev F1 | H1 there | best epoch | mean last 5 F1 | best dev H1 | status |", file=sys.stderr)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|", file=sys.stderr)
for r in rows:
    vals = (f"{r['best_dev_f1']:.4f} | {r['h1_at_best_f1']:.4f} | {r['best_epoch']} | {r['mean_last5_dev_f1']:.4f} | {r['best_dev_h1']:.4f}"
            if "best_dev_f1" in r else "- | - | - | - | -")
    status = "SELECTED" if r is best else ("ok" if r["trained"] else "FAILED/incomplete")
    print(f"| {r['run']} | {r['num_gnn']} | {r['num_iter']} | {r['entity_dim']} | {vals} | {status} |", file=sys.stderr)
print(best["run"])
PY
)"
        echo "Selected on dev F1: $BEST"
        [[ -s "$PROJECT/rankings/${BEST}_test.jsonl" ]] || RUN_ID="$BEST" bash "$0" eval
        [[ -s "$PROJECT/reports/${BEST}_vs_dense.json" ]] || RUN_ID="$BEST" bash "$0" score
        echo "Final comparison: reports/${BEST}_vs_dense.json and reports/${BEST}_vs_dense_ci.json"
        ;;
    *) echo "Unknown stage: $STAGE"; exit 1 ;;
esac
