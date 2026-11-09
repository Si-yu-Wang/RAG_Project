# 运行 GNN-RAG（ReaRev）与 Dense 对比

三个数据集各有一个入口脚本：`run_webqsp.sh`、`run_qasper.sh`、`run_pubmedqa.sh`。每个脚本按阶段执行，已有的输出不会被覆盖；中断后，重新运行同一阶段即可。结果见 [`docs/experiments/gnn_rag/`](../../docs/experiments/gnn_rag/README.md)。

## 1. 依赖（三个数据集共用）

先按本机 CUDA 版本安装 PyTorch 2.x，再安装其余依赖：

```bash
python -m pip install -r data_process/gnn_runs/requirements.txt
```

GNN-RAG 官方源码固定在 commit `28d31bc0`（https://github.com/cmavro/GNN-RAG），放在项目同级目录 `../GNN-RAG/gnn`，然后打补丁：

```bash
python data_process/gnn_runs/prepare_gnn.py ../GNN-RAG/gnn
```

`prepare_gnn.py` 只做三处修改：修掉启动时的 NameError、去掉作者机器上硬编码的模型缓存路径、关闭训练中的自动 test 评测（训练只看 dev，test 由 `eval` 阶段单独运行）。ReaRev 的网络、损失和评分都没有改动。

QASPER 和 WebQSP 的正式 Dense 使用 Member 2 的原实现，需要另建环境（在 `baselines/member2_dense_rag/` 下按其 README 安装）。PubMedQA 的 Dense 直接使用同一 BGE 模型的向量，不需要另建环境。

## 2. 三个数据集的运行顺序

| 数据集 | 命令（在项目根目录执行） | 说明 |
|---|---|---|
| WebQSP | `run_webqsp.sh smoke` → `train` → `eval` → `dense` → `score` | 使用 GNN-RAG 原生子图 `data/data/webqsp`；ReaRev 训练 200 轮，按 dev H1 选 checkpoint |
| QASPER | `run_qasper.sh export` → `embed-chunks` → `embed-queries` → `build` → `smoke` → `dense` → `floors` → `tune` | `tune` 训练 9 组配置（每组 30 轮），按留出 dev H1 选出一组，只把这一组在 test 上评测并与 Dense 对比 |
| PubMedQA | `run_pubmedqa.sh all` | 依次完成编码、建图、冒烟、Dense/BM25、训练（100 轮）、推理、评分；按 dev F1 选 checkpoint |

调用方式统一为 `bash data_process/gnn_runs/<脚本> <阶段>`。用 `RUN_ID=...` 区分不同的 run；超参数可用环境变量覆盖，例如 `ENTITY_DIM`、`NUM_GNN`、`NUM_ITER`、`EPOCHS`。训练时超参数会写入 `runs/<RUN_ID>/hparams.env`，评测时自动读取，保证模型结构一致。

## 3. 输出

| 位置 | 内容 |
|---|---|
| `runs/<RUN_ID>/` | checkpoint、训练日志、命令、超参数、环境记录（不入库） |
| `rankings/*.jsonl` | 每个方法的逐题排名（不入库） |
| `reports/*_vs_dense.json` | 各方法主表 |
| `reports/*_vs_dense_ci.json` | ReaRev − Dense 的配对 bootstrap 95% CI（10,000 次） |

## 4. 关键约定

- 选 checkpoint 和调参只用 dev，test 只评测选中的模型一次。
- GNN 评测使用 `--eps 2` 导出完整候选排名，再由统一评分器计算 Hit/Recall/MRR@1/3/5/10。不使用 GNN-RAG 自带的 hits/F1，因为它的分母和候选过滤方式不同。
- ReaRev 为有监督训练，Dense（BGE-small）为零样本，报告中需注明。
