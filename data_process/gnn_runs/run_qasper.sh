#!/usr/bin/env bash
# QASPER: encode -> build ReaRev graph -> train / tune -> eval -> Dense and floors -> score.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GNN_DIR="${GNN_DIR:-$(dirname -- "$PROJECT")/GNN-RAG/gnn}"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
STAGE="${1:?Usage: run_qasper.sh export|embed-chunks|embed-queries|build|smoke|train|eval|dense|floors|score|tune}"
DEFAULT_RUN=qasper_rearev_s1
[[ "$STAGE" != smoke ]] || DEFAULT_RUN=qasper_rearev_smoke
RUN_ID="${RUN_ID:-$DEFAULT_RUN}"
[[ "$RUN_ID" =~ ^qasper_[A-Za-z0-9_-]+$ ]] || { echo 'RUN_ID must start with qasper_'; exit 1; }
RUN_DIR="$PROJECT/runs/$RUN_ID"
PREP="$PROJECT/runs/qasper_prepare"
GRAPH="$PROJECT/data/processed/qasper_gnn"
SUPPORT="$PROJECT/data/processed/qasper_support"
CHUNKS="$PROJECT/data/processed/chunks/size512_ov102_soft/chunks.parquet"
QA="$PROJECT/data/processed/qa.parquet"
SEED="${SEED:-19960626}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
# ReaRev hyperparameters. Defaults = the original qasper_rearev_s1 configuration (copied from
# the WebQSP/ReaRev KGQA setup). `train` saves them to runs/<RUN_ID>/hparams.env; every later
# stage on that run reloads them, so eval always rebuilds the exact trained architecture.
ENTITY_DIM="${ENTITY_DIM:-50}"
NUM_ITER="${NUM_ITER:-2}"
NUM_INS="${NUM_INS:-3}"
NUM_GNN="${NUM_GNN:-3}"
LR="${LR:-0.0005}"
if [[ "$STAGE" != train && "$STAGE" != smoke && -f "$RUN_DIR/hparams.env" ]]; then
    # shellcheck disable=SC1091
    source "$RUN_DIR/hparams.env"
    echo "Loaded $RUN_ID hparams: entity_dim=$ENTITY_DIM num_iter=$NUM_ITER num_ins=$NUM_INS num_gnn=$NUM_GNN lr=$LR seed=$SEED batch=$BATCH_SIZE"
fi
# Hyperparameter grid for QASPER (small graphs, 1 hop root->candidates, informative 384-d features).
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TUNE_GRID="${TUNE_GRID:-3:2:50 1:1:128 1:2:128 2:1:128 2:2:128 1:1:256 1:2:256 2:1:256 2:2:256}"  # num_gnn:num_iter:entity_dim
mkdir -p "$PREP" "$PROJECT/rankings" "$PROJECT/reports"

fresh() { for p in "$@"; do [[ ! -e "$p" ]] || { echo "Output exists; preserve it before retrying: $p"; exit 1; }; done; }
cuda_check() { python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones((32,32),device="cuda"); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==32; print("CUDA compute OK")'; }
logged() {
    local prefix="$1"; shift
    fresh "$prefix.command.txt" "$prefix.console.log"
    printf '%q ' "$@" > "$prefix.command.txt"
    printf '\n' >> "$prefix.command.txt"
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
    actual = hashlib.sha256((d/name).read_text(encoding='utf-8').encode()).hexdigest()
    assert actual == hashes['patched_sha256_lf'], name
print('Source patch hashes verified; train evaluates held-out dev only.')
PY
}
record_env() {
    python -m pip freeze > "$RUN_DIR/environment.txt"
    python -V > "$RUN_DIR/python-version.txt" 2>&1
    cp "$GNN_DIR/webqsp_patch_record.json" "$RUN_DIR/"
    cp "$GRAPH/manifest.json" "$RUN_DIR/graph-manifest.json"
}
gnn_args=(ReaRev --name qasper --data_folder "$GRAPH/" --lm sbert --lm_frozen 1
    --relation_word_emb False --entity_emb_file node_feat.npy --kge_frozen 1
    --entity_dim "$ENTITY_DIM" --num_iter "$NUM_ITER" --num_ins "$NUM_INS" --num_gnn "$NUM_GNN"
    --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
    --eval_every 2 --lr "$LR" --seed "$SEED" --checkpoint_dir "$RUN_DIR")

case "$STAGE" in
    export)
        fresh "$GRAPH/_embed/chunks.jsonl" "$GRAPH/_embed/queries.jsonl"
        logged "$PREP/export" python "$PROJECT/data_process/qasper_to_gnn.py" export
        ;;
    embed-chunks|embed-queries)
        cuda_check
        kind="${STAGE#embed-}"
        test -f "$GRAPH/_embed/$kind.jsonl"
        fresh "$GRAPH/_embed/$kind.npy" "$GRAPH/_embed/$kind.ids.txt"
        extra=()
        [[ "$kind" != queries ]] || extra=(--query)
        [[ "${ALLOW_DOWNLOAD:-0}" != 1 ]] || extra+=(--allow-download)
        logged "$PREP/$STAGE" python "$PROJECT/data_process/embed_bge.py" \
            --input "$GRAPH/_embed/$kind.jsonl" --out "$GRAPH/_embed/$kind" \
            --device cuda --batch-size "${EMBED_BATCH_SIZE:-32}" "${extra[@]}"
        ;;
    build)
        for name in train.json dev.json test.json entities.txt relations.txt node_feat.npy id_map.parquet manifest.json vocab.txt word_emb.npy heldout_train_papers.txt; do
            fresh "$GRAPH/$name"
        done
        test -f "$GRAPH/_embed/chunks.npy"
        test -f "$GRAPH/_embed/queries.npy"
        logged "$PREP/build" python "$PROJECT/data_process/qasper_to_gnn.py" build --webqsp-dir "$SUPPORT"
        python "$PROJECT/data_process/check_gnn_folder.py" "$GRAPH"
        ;;
    smoke|train)
        fresh "$RUN_DIR"
        cuda_check
        check_patch
        mkdir -p "$RUN_DIR"
        printf 'ENTITY_DIM=%q\nNUM_ITER=%q\nNUM_INS=%q\nNUM_GNN=%q\nLR=%q\nSEED=%q\nBATCH_SIZE=%q\nTEST_BATCH_SIZE=%q\n' \
            "$ENTITY_DIM" "$NUM_ITER" "$NUM_INS" "$NUM_GNN" "$LR" "$SEED" "$BATCH_SIZE" "$TEST_BATCH_SIZE" > "$RUN_DIR/hparams.env"
        record_env
        if [[ "$STAGE" == smoke ]]; then
            (cd "$GNN_DIR"; logged "$RUN_DIR/$RUN_ID" python -u main.py "${gnn_args[@]}" \
                --experiment_name "$RUN_ID" --num_epoch 2 --max_train 64)
            test -s "$RUN_DIR/$RUN_ID-final.ckpt"
            echo 'Smoke passed: ReaRev loads the QASPER graph, trains 2 epochs and saves a checkpoint.'
        else
            (cd "$GNN_DIR"; logged "$RUN_DIR/$RUN_ID" python -u main.py "${gnn_args[@]}" \
                --experiment_name "$RUN_ID" --num_epoch "${EPOCHS:-50}")
            test -s "$RUN_DIR/$RUN_ID-h1.ckpt"
            sha256sum "$RUN_DIR/$RUN_ID-h1.ckpt" > "$RUN_DIR/selected_checkpoint.sha256"
            echo 'Training complete. Check logs and use dev-H1 checkpoint for explicit eval.'
        fi
        ;;
    eval)
        cuda_check
        check_patch
        CKPT="${CKPT:-$RUN_ID-h1.ckpt}"
        test -s "$RUN_DIR/$CKPT"
        fresh "$PROJECT/rankings/${RUN_ID}_qasper_dev.jsonl" "$PROJECT/reports/${RUN_ID}_qasper_dev.json"
        (cd "$GNN_DIR"; logged "$RUN_DIR/${RUN_ID}_eval" python -u main.py "${gnn_args[@]}" \
            --experiment_name "${RUN_ID}_eval" --is_eval --load_experiment "$CKPT" --eps 2)
        python "$PROJECT/data_process/score_retrieval.py" from-gnn --mode qasper --gnn-dir "$GRAPH" \
            --info "$RUN_DIR/${RUN_ID}_eval_test.info" --method "$RUN_ID" \
            --out "$PROJECT/rankings/${RUN_ID}_qasper_dev.jsonl"
        python "$PROJECT/data_process/score_retrieval.py" score --dataset qasper --split dev \
            "$PROJECT/rankings/${RUN_ID}_qasper_dev.jsonl" --out "$PROJECT/reports/${RUN_ID}_qasper_dev.json"
        ;;
    dense)
        cuda_check
        ART="$PROJECT/artifacts/qasper_dev_member2"
        fresh "$ART" "$PROJECT/rankings/qasper_dev_dense_member2.jsonl"
        mkdir -p "$ART"
        python -m pip freeze > "$ART/environment.txt"
        (
            cd "$PROJECT/baselines/member2_dense_rag"
            logged "$ART/export" python -m dense_rag.evaluation export --qa "$QA" --split dev --output "$ART/requests.jsonl"
            logged "$ART/build" python -m dense_rag.cli build --chunks "$CHUNKS" --split dev --device cuda --index "$ART/index"
            logged "$ART/retrieve" python -m dense_rag.cli run --index "$ART/index" --questions "$ART/requests.jsonl" \
                --split dev --scope paper --top-k 10 --device cuda --generator preview --output "$ART/predictions.jsonl"
            logged "$ART/metrics" python -m dense_rag.evaluation score --qa "$QA" --predictions "$ART/predictions.jsonl" \
                --index "$ART/index" --split dev --ks 1 3 5 10 --output "$ART/member2_metrics.json"
        )
        python "$PROJECT/data_process/score_retrieval.py" from-dense --predictions "$ART/predictions.jsonl" \
            --method dense_bge_member2 --out "$PROJECT/rankings/qasper_dev_dense_member2.jsonl"
        ;;
    floors)
        for kind in order bm25; do fresh "$PROJECT/rankings/qasper_dev_$kind.jsonl"; done
        for kind in order bm25; do
            python "$PROJECT/data_process/score_retrieval.py" baseline --dataset qasper --split dev \
                --kind "$kind" --out "$PROJECT/rankings/qasper_dev_$kind.jsonl"
        done
        ;;
    score)
        GNN_RANK="$PROJECT/rankings/${RUN_ID}_qasper_dev.jsonl"
        DENSE_RANK="$PROJECT/rankings/qasper_dev_dense_member2.jsonl"
        test -s "$GNN_RANK"
        test -s "$DENSE_RANK"
        for kind in order bm25; do test -s "$PROJECT/rankings/qasper_dev_$kind.jsonl"; done
        fresh "$PROJECT/reports/${RUN_ID}_vs_dense.json" "$PROJECT/reports/${RUN_ID}_vs_dense_ci.json"
        python "$PROJECT/data_process/score_retrieval.py" score --dataset qasper --split dev \
            "$DENSE_RANK" "$GNN_RANK" "$PROJECT/rankings/qasper_dev_order.jsonl" "$PROJECT/rankings/qasper_dev_bm25.jsonl" \
            --random-floor 50 --out "$PROJECT/reports/${RUN_ID}_vs_dense.json"
        python "$PROJECT/data_process/score_retrieval.py" compare --dataset qasper --split dev \
            "$DENSE_RANK" "$GNN_RANK" --n-boot 10000 --out "$PROJECT/reports/${RUN_ID}_vs_dense_ci.json"
        ;;
    tune)
        # Train every grid config on the same data/seed/budget, pick by held-out dev H1 (never test),
        # then evaluate ONLY the selected config on the original QASPER dev and compare with Dense.
        test -s "$PROJECT/rankings/qasper_dev_dense_member2.jsonl" || { echo 'Run the dense stage first.'; exit 1; }
        for spec in $TUNE_GRID; do
            IFS=: read -r g i d <<< "$spec"
            run="qasper_tune_g${g}_i${i}_d${d}"
            if [[ -s "$PROJECT/runs/$run/selected_checkpoint.sha256" ]]; then echo "skip (trained): $run"; continue; fi
            if [[ -e "$PROJECT/runs/$run" ]]; then echo "SKIP incomplete/failed run (move it aside to retry): runs/$run"; continue; fi
            echo "=== train $run (num_gnn=$g num_iter=$i entity_dim=$d, $TUNE_EPOCHS epochs) ==="
            RUN_ID="$run" EPOCHS="$TUNE_EPOCHS" NUM_GNN="$g" NUM_ITER="$i" ENTITY_DIM="$d" bash "$0" train                 || echo "FAILED: $run (continuing with the rest of the grid)"
        done
        SUMMARY="$PROJECT/reports/qasper_tune_summary.json"
        BEST="$(python - "$PROJECT/runs" "$SUMMARY" $TUNE_GRID <<'PY'
import json, pathlib, re, sys
runs, out, specs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
pat = re.compile(r"EVAL F1: ([0-9.]+), H1: ([0-9.]+)")
rows = []
for spec in specs:
    g, i, d = spec.split(":")
    run = f"qasper_tune_g{g}_i{i}_d{d}"
    folder = runs / run
    # ReaRev's own log and our tee'd console log; use whichever holds the most eval lines (no double count).
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.glob("*.log"))] if folder.exists() else []
    evals = max(([(float(f1), float(h1)) for f1, h1 in pat.findall(t)] for t in texts), key=len, default=[])
    done = (folder / "selected_checkpoint.sha256").exists()
    row = {"run": run, "num_gnn": int(g), "num_iter": int(i), "entity_dim": int(d), "trained": done,
           "evals": len(evals)}
    if evals:
        k = max(range(len(evals)), key=lambda j: (evals[j][1], -j))
        row.update(best_dev_h1=evals[k][1], best_epoch=2 * (k + 1), last_dev_h1=evals[-1][1],
                   mean_last5_dev_h1=sum(e[1] for e in evals[-5:]) / len(evals[-5:]))
    rows.append(row)
ok = [r for r in rows if r["trained"] and "best_dev_h1" in r]
if not ok:
    sys.exit("No completed tuning run")
best = max(ok, key=lambda r: (r["best_dev_h1"], -r["entity_dim"], -r["num_gnn"], -r["num_iter"]))
out.write_text(json.dumps({"selection": "max held-out dev H1 (ReaRev native, train-held-out papers); test not used",
                           "selected": best["run"], "runs": rows}, indent=2), encoding="utf-8")
print("| run | gnn | iter | dim | best dev H1 | best epoch | mean last 5 | status |", file=sys.stderr)
print("|---|---:|---:|---:|---:|---:|---:|---|", file=sys.stderr)
for r in rows:
    status = "SELECTED" if r is best else ("ok" if r["trained"] else "FAILED/incomplete")
    vals = (f"{r['best_dev_h1']:.4f} | {r['best_epoch']} | {r['mean_last5_dev_h1']:.4f}"
            if "best_dev_h1" in r else "- | - | -")
    print(f"| {r['run']} | {r['num_gnn']} | {r['num_iter']} | {r['entity_dim']} | {vals} | {status} |", file=sys.stderr)
print(best["run"])
PY
)"
        echo "Selected on held-out dev: $BEST (summary: $SUMMARY)"
        if [[ ! -s "$PROJECT/rankings/${BEST}_qasper_dev.jsonl" ]]; then RUN_ID="$BEST" bash "$0" eval; fi
        if [[ ! -s "$PROJECT/reports/${BEST}_vs_dense.json" ]]; then
            if [[ ! -e "$PROJECT/rankings/qasper_dev_order.jsonl" && ! -e "$PROJECT/rankings/qasper_dev_bm25.jsonl" ]]; then
                RUN_ID="$BEST" bash "$0" floors
            fi
            RUN_ID="$BEST" bash "$0" score
        fi
        echo "Final comparison: reports/${BEST}_vs_dense.json and reports/${BEST}_vs_dense_ci.json"
        ;;
    *) echo "Unknown stage: $STAGE"; exit 1 ;;
esac
