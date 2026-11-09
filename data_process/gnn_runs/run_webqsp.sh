#!/usr/bin/env bash
# Run one explicit stage; no environment installation or data downloads here.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GNN_DIR="${GNN_DIR:-$(dirname -- "$PROJECT")/GNN-RAG/gnn}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$(dirname -- "$PROJECT")/hf-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
STAGE="${1:?Usage: bash run_webqsp.sh smoke|train|eval|dense|dense-ref|score}"
RUN_ID="${RUN_ID:-webqsp_rearev_s1}"
SEED="${SEED:-19960626}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID'; exit 1; }
RUN_DIR="$PROJECT/runs/$RUN_ID"
RAW="$PROJECT/data/data/webqsp"
TEXT="$PROJECT/data/processed/webqsp_text/test"
mkdir -p "$RUN_DIR" "$PROJECT/rankings" "$PROJECT/reports"

record_env() {
    python -m pip freeze > "$RUN_DIR/environment.txt"
    python -V > "$RUN_DIR/python-version.txt" 2>&1
    nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
    cp "$GNN_DIR/webqsp_patch_record.json" "$RUN_DIR/"
    cp "$TEXT/manifest.json" "$RUN_DIR/text-manifest.json"
}

gnn_run() {
    local name="$1"; shift
    local cmd=(python -u main.py ReaRev
        --name webqsp --data_folder "$RAW/"
        --lm sbert --lm_frozen 1 --relation_word_emb True
        --entity_dim 50 --num_iter 3 --num_ins 2 --num_gnn 3
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --eval_every 2 --lr 0.0005 --seed "$SEED"
        --checkpoint_dir "$RUN_DIR" --experiment_name "$name" "$@")
    (cd "$GNN_DIR"; printf '%q ' "${cmd[@]}"; printf '\n') > "$RUN_DIR/$name.command.txt"
    (cd "$GNN_DIR"; "${cmd[@]}") 2>&1 | tee "$RUN_DIR/$name.console.log"
}

case "$STAGE" in
    smoke|train)
        [[ ! -e "$RUN_DIR/$RUN_ID.command.txt" ]] || { echo 'RUN_ID already used; choose a new one.'; exit 1; }
        [[ -f "$GNN_DIR/webqsp_patch_record.json" ]] || { echo 'Run prepare_gnn.py first'; exit 1; }
        python "$PROJECT/data_process/gnn_runs/verify_webqsp.py"
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
        record_env
        if [[ "$STAGE" == smoke ]]; then
            gnn_run "$RUN_ID" --num_epoch 2 --max_train 64
            test -f "$RUN_DIR/$RUN_ID-final.ckpt"
            python -c 'import sys, torch; c = torch.load(sys.argv[1], map_location="cpu"); assert c["model_state_dict"]; print("Checkpoint serialization readable")' "$RUN_DIR/$RUN_ID-final.ckpt"
            echo 'Smoke train/dev and checkpoint serialization checks complete; reserve test for the final selected model.'
        else
            gnn_run "$RUN_ID" --num_epoch "$EPOCHS"
            test -f "$RUN_DIR/$RUN_ID-h1.ckpt"
            echo 'Training complete. Select dev H1 checkpoint; run eval explicitly.'
        fi
        ;;
    eval)
        test -f "$GNN_DIR/webqsp_patch_record.json"
        python "$PROJECT/data_process/gnn_runs/verify_webqsp.py"
        CKPT="${CKPT:-$RUN_ID-h1.ckpt}"
        test -f "$RUN_DIR/$CKPT"
        [[ ! -e "$RUN_DIR/${RUN_ID}_eval.command.txt" ]] || { echo 'Evaluation output exists; preserve it before rerunning.'; exit 1; }
        gnn_run "${RUN_ID}_eval" --is_eval --load_experiment "$CKPT" --eps 2
        python "$PROJECT/data_process/score_retrieval.py" from-gnn --mode webqsp \
            --gnn-dir "$RAW" --info "$RUN_DIR/${RUN_ID}_eval_test.info" \
            --text-dir "$TEXT" --method "$RUN_ID" --out "$PROJECT/rankings/${RUN_ID}_test.jsonl"
        python "$PROJECT/data_process/score_retrieval.py" score --dataset webqsp --split test \
            "$PROJECT/rankings/${RUN_ID}_test.jsonl" --out "$PROJECT/reports/${RUN_ID}_test.json"
        ;;
    dense)
        [[ ! -e "$PROJECT/rankings/webqsp_dense_test.jsonl" ]] || { echo 'Dense rankings already exist'; exit 1; }
        ART="$PROJECT/artifacts/webqsp_test_member2"
        [[ ! -e "$ART" ]] || { echo 'Dense artifact directory exists; preserve it before retrying.'; exit 1; }
        python "$PROJECT/data_process/gnn_runs/verify_webqsp.py" --text-only
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
        mkdir -p "$ART"
        python -m pip freeze > "$ART/environment.txt"
        (
            cd "$PROJECT/baselines/member2_dense_rag"
            python -m dense_rag.evaluation export --qa "$TEXT/qa.parquet" --split test --output "$ART/requests.jsonl"
            python -m dense_rag.cli build --chunks "$TEXT/chunks_slim.parquet" --split test --device cuda --index "$ART/index"
            python -m dense_rag.cli run --index "$ART/index" --questions "$ART/requests.jsonl" \
                --split test --scope paper --top-k 10 --device cuda --generator preview --output "$ART/predictions.jsonl"
            python -m dense_rag.evaluation score --qa "$TEXT/qa.parquet" --predictions "$ART/predictions.jsonl" \
                --index "$ART/index" --split test --ks 1 3 5 10 --output "$ART/member2_metrics.json"
        ) 2>&1 | tee "$ART/console.log"
        python "$PROJECT/data_process/score_retrieval.py" from-dense --predictions "$ART/predictions.jsonl" \
            --method dense_bge_member2 --out "$PROJECT/rankings/webqsp_dense_test.jsonl"
        ;;
    dense-ref)
        [[ ! -e "$PROJECT/rankings/webqsp_dense_ref_test.jsonl" ]] || { echo 'Reference Dense rankings already exist'; exit 1; }
        python "$PROJECT/data_process/gnn_runs/verify_webqsp.py" --text-only
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
        python "$PROJECT/data_process/dense_rank.py" export --dataset webqsp --split test
        for kind in chunks queries; do
            extra=()
            [[ "$kind" != queries ]] || extra=(--query)
            python "$PROJECT/data_process/embed_bge.py" --input "$TEXT/_embed/$kind.jsonl" \
                --out "$TEXT/_embed/$kind" --device cuda --batch-size 64 "${extra[@]}"
        done
        python "$PROJECT/data_process/dense_rank.py" rank --dataset webqsp --split test \
            --depth 100 --method dense_bge_ref --out "$PROJECT/rankings/webqsp_dense_ref_test.jsonl"
        ;;
    score)
        GNN_RANK="$PROJECT/rankings/${RUN_ID}_test.jsonl"
        DENSE_RANK="${DENSE_RANK:-$PROJECT/rankings/webqsp_dense_test.jsonl}"
        test -f "$GNN_RANK"
        test -f "$DENSE_RANK"
        for suffix in vs_dense vs_dense_1628 vs_dense_ci; do
            [[ ! -e "$PROJECT/reports/${RUN_ID}_${suffix}.json" ]] || { echo 'Report exists; preserve it before rerunning.'; exit 1; }
        done
        python "$PROJECT/data_process/score_retrieval.py" score --dataset webqsp --split test \
            "$DENSE_RANK" "$GNN_RANK" --random-floor 20 \
            --out "$PROJECT/reports/${RUN_ID}_vs_dense.json"
        python "$PROJECT/data_process/score_retrieval.py" score --dataset webqsp --split test \
            "$DENSE_RANK" "$GNN_RANK" --count-as-miss answer_not_in_subgraph \
            --out "$PROJECT/reports/${RUN_ID}_vs_dense_1628.json"
        python "$PROJECT/data_process/score_retrieval.py" compare --dataset webqsp --split test \
            "$DENSE_RANK" "$GNN_RANK" --out "$PROJECT/reports/${RUN_ID}_vs_dense_ci.json"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
