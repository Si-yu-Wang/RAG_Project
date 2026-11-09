# data_process：WebQSP ⇄ 文本 RAG 格式的双向转换与统一检索评测

目标：让 **Dense RAG（member 2）** 和 **GNN-RAG（ReaRev）** 在同一个数据集上 **排同一批对象、用同一个 gold、同一组 K、同一个分母、同一个评分脚本**，只比检索准确率（不做 LLM 生成）。

| 方向 | 输入 | 输出 | 用途 |
|---|---|---|---|
| WebQSP → 文本格式 | `data/data/webqsp`（稠密 NSM 子图，GNN-RAG 自己的候选池） | `data/processed/webqsp_text/<split>/` | 给 dense baseline 跑 |
| QASPER → WebQSP 格式 | `data/processed/chunks/size512_ov102_soft/` + `qa_with_gold.jsonl` | `data/processed/qasper_gnn/` | 给 GNN-RAG 的 ReaRev 跑 |
| GNN-RAG 原生 | `data/data/webqsp`（不需要转换） | — | WebQSP 上的 GNN |
| PubMedQA 引文图 → WebQSP 格式 | `data/processed/pubmedqa_graph/`（组员建的 1,395 篇引文语料） | `data/processed/pubmedqa_gnn/` | 给 ReaRev 跑；Dense / BM25 / 评分也在同一脚本（`pubmedqa_to_gnn.py`，见 §4b） |

---

## 1. 统一的评测口径（两个方向共用）

所有方法最后都变成同一种东西：**对某个问题候选池里的 chunk 做排序**（`ranked_chunk_ids`），由 `score_retrieval.py` 统一打分。指标定义与 member 2 的 `dense_rag/retrieval_metrics.py` 完全一致（段落级）：

| 指标 | 定义 |
|---|---|
| Hit@K | top-K chunk 覆盖的段落里有任一 gold 段落 → 1 |
| Recall@K | \|gold ∩ top-K chunk 的 para_ids 并集\| / \|gold\|，按题宏平均 |
| MRR@K | 第一个覆盖 gold 段落的 chunk 的排名倒数，top-K 内没有则 0 |

| 数据集 | 排序单位 | 候选池 | gold | 主分母 |
|---|---|---|---|---|
| WebQSP（test） | 1 个候选节点 = 1 个段落 = 1 个 chunk | 本题子图中非 topic、非 CVT 节点（每题约 1,113 个） | 答案节点对应的单元 | 1,551（`zero_gold_reason == ""`）；另报 1,628：`--count-as-miss answer_not_in_subgraph` |
| QASPER（dev） | chunk（展开成段落） | 本篇论文全部 chunk（= dense 的 `scope=paper`） | gold 段落 | 890 |

- K 统一用 1/3/5/10。QASPER 的 @10 基本饱和（随机就有 86%），主看 @1/@3/@5 和 MRR。
- GNN 必须用 `--eps 2` 评测，才会输出完整排序（默认 0.95 会截断候选，K>1 只能当下限）。**GNN-RAG 自己打印的 hits/F1 不要用**：它把无答案题算命中、只看 top-1。
- 每组都报下限（`baseline` 子命令：random / order / degree / bm25）和配对 bootstrap 置信区间（`compare` 子命令），并注明 GNN 是有监督训练、dense 是零样本。

排序文件格式（JSON lines，任何新方法照此输出即可接入）：

```json
{"question_id": "WebQTest-0", "method": "gnn_rearev", "ranked_chunk_ids": ["...", "..."], "scores": [0.91, 0.03]}
```

---

## 2. 环境

- 转换和评分：仓库默认 Python（3.11+，`numpy`、`pyarrow`、`tokenizers`）。
- BGE 向量（`embed_bge.py`）：任何装了 `torch` + `transformers` 的环境（不需要 sentence-transformers / faiss），GPU 更快。配置与 dense baseline 完全一致：`BAAI/bge-small-en-v1.5@5c38ec7c`、CLS 池化、L2 归一化、max_length 512、同一个查询前缀。
- GNN-RAG 本身需要单独环境（`transformers<5`，dense 环境锁的是 5.x），并修掉 `gnn/parsing.py:81-82` 的 NameError；运行时用 `--lm sbert`、设 `PYTHONUTF8=1`。这些是代码/环境问题，不在本目录处理。

---

## 3. WebQSP → 文本格式（给 dense 跑）

```bash
# 0) 取实体名数据（GNN-RAG entities_names.json + HF RoG-webqsp），按 sha256 校验
python data_process/fetch_webqsp_names.py
# 1) 合并成全局 MID→名称表（RoG 按位置对齐；0 冲突）→ data/raw/webqsp_names/mid2name.tsv
python data_process/build_mid2name.py
# 2) 转换（默认 dev + test；train 只在要微调 BGE 时才需要：--splits train dev test）
python data_process/webqsp_to_text.py
```

设计（`webqsp_u3path_v1`）：

- **文档**（`paper_id`）：每个 (split, topic 集合) 一篇，`wqsp-<split>-<排序后的 topic MID 用 + 连接>`。同一 topic 集合的子图完全相同（逐篇用 tuple 指纹校验）。
- **单元**：每个候选节点一条。文本是从 topic 到该节点的最多 3 条最短路径，一行一条，CVT 两跳合并，例如
  `Eli Whitney | law.inventor.inventions / ~law.invention.inventor | Cotton gin`
  `~r` 表示逆着边的方向走；没有名字的非 CVT 节点写原始 MID（RoG / GNN-RAG 的惯例）。
- **CVT 规则**：没有名字、在子图里有出边、且不是 topic 的 MID。答案从来不是 CVT。
- **实体名**：只用全局名表（`entities_names.json` ∪ RoG 对齐），**绝不用 `answers[].text`**（那只给答案起名，等于泄题）。
- **顺序与选择**：路径、关系标签、CVT 属性和单元顺序都用与标签无关的确定性排序；答案只在最后用来写 gold 列。
- **`zero_gold_reason`**：`""` / `no_answer` / `answer_not_in_subgraph` / `answer_only_topic_entity` / `answer_cvt_node`。
- 空子图的 2 道题各放 1 个占位单元，否则 dense 的评测会因 `no_evidence` 中止。

每个 split 目录下的文件：

| 文件 | 说明 |
|---|---|
| `chunks.parquet` | 契约的 23 个字段（`para_ids`/`para_spans` 1:1，`config_hash` 与 manifest 一致） |
| `chunks_slim.parquet` | **给 dense 建索引用**（8 列，省内存：test 有 105 万行） |
| `paragraphs.parquet` | 契约的 14 个字段 |
| `qa.parquet` / `qa_with_gold.jsonl` | 契约的 24 个字段 + `answer_kb_ids`、`topic_mids` |
| `unit_nodes.parquet` | 无标签旁表：chunk ↔ 节点（用于把 GNN 的节点排序映射回 chunk）、hop、是否有名字、度数 |
| `manifest.json` | 配置、config_hash、统计、各文件 sha256 |

规模（`webqsp_text/summary.json`）：

| split | 题 | 文档 | 单元 | token 总数 | 可评题 | gold/题 | 完美排序上限 R@1/3/5/10 |
|---|---|---|---|---|---|---|---|
| dev | 250 | 226 | 237,290 | 11.7M | 235 | 5.89 | 61.51 / 83.26 / 89.34 / 94.61 |
| test | 1,639 | 1,057 | 1,052,134 | 52.5M | 1,551 | 5.62 | 63.11 / 83.79 / 89.70 / 94.24 |

（多 gold 的题 Recall@1 不可能是 1，所以上限 < 100；Hit@K 的上限是 100。）

跑 dense（member 2 的环境，在 `baselines/member2_dense_rag/` 下）：

```powershell
$W = "D:/project/6405/RAG_Project/data/processed/webqsp_text/test"
.venv/Scripts/python -m dense_rag.evaluation export --qa "$W/qa.parquet" --split test --output artifacts/wq_test_requests.jsonl
.venv/Scripts/python -m dense_rag.cli build --chunks "$W/chunks_slim.parquet" --split test --index artifacts/wq_test_index
.venv/Scripts/python -m dense_rag.cli run --index artifacts/wq_test_index --questions artifacts/wq_test_requests.jsonl --split test --scope paper --top-k 10 --output artifacts/wq_test_predictions.jsonl
```

test 索引约 105 万个短单元，建索引时内存峰值约 6.5–7 GB，建议先跑 dev、单独跑。也可以用本目录的参考实现在 GPU 上算（见 §5）。

member 2 自带的严格评分器也能直接读这里的 `qa.parquet`，可以拿来交叉核对（只报 Recall/MRR，分母同样是 1,551）：

```powershell
.venv/Scripts/python -m dense_rag.evaluation score --qa "$W/qa.parquet" --predictions artifacts/wq_test_predictions.jsonl --index artifacts/wq_test_index --split test --ks 1 3 5 10 --output artifacts/wq_test_metrics.json
```

---

## 4. QASPER → WebQSP 格式（给 GNN-RAG 跑）

```bash
python data_process/qasper_to_gnn.py export
# 在有 torch 的环境里（例如 GPU 环境）算 BGE 向量
python data_process/embed_bge.py --input data/processed/qasper_gnn/_embed/queries.jsonl --out data/processed/qasper_gnn/_embed/queries --query
python data_process/embed_bge.py --input data/processed/qasper_gnn/_embed/chunks.jsonl  --out data/processed/qasper_gnn/_embed/chunks
python data_process/qasper_to_gnn.py build
python data_process/check_gnn_folder.py data/processed/qasper_gnn
```

设计（`qasper_inpaper_s1qc_v1`）：

- 每道题一张图：一个虚拟问题根节点 `Q::<qid>`（作为 topic entity / 种子）+ 该论文每个 chunk 的一个实例节点 `<qid>|<chunk_id>`。候选池与 dense 的篇内检索完全相同。
- 关系（`relations.txt` 的顺序即 id）：`0 query.chunk.candidate`（根→每个 chunk）、`1 paper.chunk.next_same_section`、`2 paper.chunk.next_new_section`、`3 paper.chunk.similar`（论文内 BGE kNN，k=3，去掉已是相邻的对）。ReaRev 自己会加逆向边，所以每条边只存一次。
- **节点特征**：`node_feat.npy`，chunk 实例行 = BGE(q) ⊙ BGE(c)（逐元素乘，求和就是 dense 的余弦），根节点行 = 0。ReaRev 本身看不到节点文本，不给特征就只能看边的类型，结果约等于随机。
- 答案 = `gold_chunk_ids`（写成实体 key 字符串）。
- 划分：`train.json` = QASPER-train 中可评题（去掉留出论文）；`dev.json` = 留出的 15% train 论文上的可评题（选 checkpoint 用）；`test.json` = QASPER-dev 全部 1,005 题（打分只算 890 道可评题，与 dense 同分母）。**不要在 test.json 上选模型。**
- 消融开关：`--no-features`（无节点特征）、`--no-knn`（只有根边和相邻边）。

规模（`qasper_gnn/manifest.json`，config_hash `bd94340c8bf7`）：

| 文件 | 题 | 论文 | 每题节点（均值/中位数/最大，含根） | 每题边（均值） | 每题答案（均值） |
|---|---|---|---|---|---|
| train.json | 1,764 | 698 | 16.25 / 15 / 97 | 48.9 | 2.04 |
| dev.json | 337 | 126 | 17.45 / 17 / 59 | 53.2 | 1.97 |
| test.json | 1,005（890 可评） | 281 | 14.79 / 14 / 63 | 43.7 | 2.34 |

全局 49,414 个节点（3,106 个根 + 46,308 个 chunk 实例），相邻边 16,484 对、kNN 边 22,350 对，`node_feat.npy` 为 49,414×384 float32（76 MB）。

GNN-RAG 训练与评测（在 GNN-RAG 的 `gnn/` 目录、单独环境中）：

```bash
python main.py ReaRev --name qasper --data_folder <绝对路径>/data/processed/qasper_gnn/ --lm sbert \
  --relation_word_emb False --entity_emb_file node_feat.npy --kge_frozen 1 \
  --entity_dim 50 --num_iter 2 --num_ins 3 --num_gnn 3 --batch_size 8 --num_epoch 50 --eval_every 2 \
  --checkpoint_dir <已存在的目录> --experiment_name qasper_s1qc
python main.py ReaRev <同上参数> --is_eval --load_experiment qasper_s1qc-h1.ckpt --eps 2 --experiment_name qasper_s1qc_eval
```

注意：`--data_folder` 末尾要有 `/`，路径里不能出现 `sr-` 或 `metaqa`；不要传 `--relation_emb_file`（ReaRev 会崩）；`--name` 不要用 `cwq`。

正式结果使用的超参数与上面示例不同（示例是最初照搬 WebQSP 的配置）：QASPER 最终选用 `--num_gnn 2 --num_iter 1 --entity_dim 256`（30 轮，按留出 dev 选），见 [`gnn_runs/README.md`](gnn_runs/README.md) 的 `tune` 阶段。

---

## 4b. PubMedQA 引文图 → WebQSP 格式（给 GNN-RAG 跑）

`pubmedqa_to_gnn.py` 一个脚本包含导出、建图、Dense、BM25、GNN 结果回写、评分和配对 bootstrap：

```bash
python data_process/pubmedqa_to_gnn.py export
python data_process/embed_bge.py --input data/processed/pubmedqa_gnn/_embed/chunks.jsonl  --out data/processed/pubmedqa_gnn/_embed/chunks
python data_process/embed_bge.py --input data/processed/pubmedqa_gnn/_embed/queries.jsonl --out data/processed/pubmedqa_gnn/_embed/queries --query
python data_process/pubmedqa_to_gnn.py build
python data_process/pubmedqa_to_gnn.py dense --out rankings/pubmedqa_test_dense_bge.jsonl
python data_process/pubmedqa_to_gnn.py bm25  --out rankings/pubmedqa_test_bm25.jsonl
python data_process/pubmedqa_to_gnn.py from-gnn --info <ckpt_dir>/<exp>_eval_test.info --dense rankings/pubmedqa_test_dense_bge.jsonl --out rankings/pubmedqa_test_gnn.jsonl
python data_process/pubmedqa_to_gnn.py score rankings/pubmedqa_test_*.jsonl
python data_process/pubmedqa_to_gnn.py compare rankings/pubmedqa_test_dense_bge.jsonl rankings/pubmedqa_test_gnn.jsonl
```

设计（`pubmedqa_citegraph_s1_v1`）：

- **任务**：跨文档检索。候选池为全部 3,707 个 chunk，gold 为出处论文的全部 chunk（与 `scripts/21_build_pubmedqa_graph.py` 的设定一致）。
- **划分**：原数据没有划分，按题固定 seed（20261109）切成 train 571 / dev 95 / test 285；dev 用于选 checkpoint，所有方法都在 test 上报告。
- **每题子图**：取 Dense 前 20 个 chunk 所在的论文，加上这些论文的 1 跳引用邻居，候选为这些论文的全部 chunk（上限 150 个）。关系：`0` 根→候选、`1` 同篇相邻、`2` 引用论文之间、`3` 候选间 BGE kNN（k=3）。节点特征与 QASPER 相同（BGE(q)⊙BGE(c)）。原始数据中 2 条论文自引用边被过滤。
- **最终排名**：候选按 ReaRev 概率排序，其余 chunk 按 Dense 排在后面，保证所有方法都对同一批 chunk 打分。
- 因为 gold 是整篇论文，chunk 级 Hit@K 与论文级 Hit@K（`paper_hit@K`）恒等。

---

## 5. 打分

```bash
# GNN 输出 → 统一排序文件
python data_process/score_retrieval.py from-gnn --mode qasper --gnn-dir data/processed/qasper_gnn --info <ckpt_dir>/qasper_s1qc_eval_test.info --out rankings/qasper_dev_gnn.jsonl
python data_process/score_retrieval.py from-gnn --mode webqsp --gnn-dir data/data/webqsp --info <ckpt_dir>/<exp>_test.info --text-dir data/processed/webqsp_text/test --out rankings/wq_test_gnn.jsonl
# member 2 的 dense 预测 → 统一排序文件
python data_process/score_retrieval.py from-dense --predictions artifacts/wq_test_predictions.jsonl --out rankings/wq_test_dense.jsonl
# 下限
python data_process/score_retrieval.py baseline --dataset webqsp --split test --kind degree --out rankings/wq_test_degree.jsonl
# 打分（多个方法一张表；--random-floor 顺带报随机下限）
python data_process/score_retrieval.py score --dataset webqsp --split test rankings/wq_test_*.jsonl --random-floor 20 --out reports/wq_test.json
python data_process/score_retrieval.py score --dataset webqsp --split test rankings/wq_test_*.jsonl --count-as-miss answer_not_in_subgraph   # 1,628 视角
python data_process/score_retrieval.py score --dataset webqsp --split test rankings/wq_test_*.jsonl --named-only                          # 名字捷径敏感性
python data_process/score_retrieval.py score --dataset qasper --split dev rankings/qasper_dev_*.jsonl
# 配对 bootstrap
python data_process/score_retrieval.py compare --dataset qasper --split dev rankings/qasper_dev_dense.jsonl rankings/qasper_dev_gnn.jsonl
```

不想装 member 2 环境时，可以用参考 dense 实现（结果与 member 2 一致，见 §6）：

```bash
python data_process/dense_rank.py export --dataset webqsp --split test
python data_process/embed_bge.py --input data/processed/webqsp_text/test/_embed/chunks.jsonl  --out data/processed/webqsp_text/test/_embed/chunks
python data_process/embed_bge.py --input data/processed/webqsp_text/test/_embed/queries.jsonl --out data/processed/webqsp_text/test/_embed/queries --query
python data_process/dense_rank.py rank --dataset webqsp --split test --out rankings/wq_test_dense_ref.jsonl
```

---

## 6. 已验证

见下方各条；复现命令都在上面。

- `build_mid2name.py`：RoG 与本地稠密子图 4,700 题逐位置对齐，关系 0 处不一致，名字 0 冲突；新增 79,931 个名字，合计 639,943 个。
- `webqsp_to_text.py`：dev/test 的文档数、可评题数、无答案/答案不在子图的题数、每题 gold 数、完美排序上限，均与先前独立分析一致；0 个指纹冲突、0 个截断单元。
- `score_retrieval.py from-gnn`：GNN-RAG 官方发布的 ReaRev 输出（`llm/results/gnn/RoG-webqsp/rearev-sbert/test.info`，eps 0.95）在 1,551 道可评题上：Hit@1 80.01、Hit@3/5/10 ≥ 87.49/90.07/92.39、Recall@1/3/5/10 ≥ 48.93/70.56/77.86/84.51、MRR@10 ≥ 0.843。
- 同一份官方输出按 1,628 道有答案题计（`--count-as-miss answer_not_in_subgraph`）：Hit@1 76.23。
- WebQSP test 下限：随机 Hit@1/Hit@10 = 0.58/5.04；度数 8.25/32.82；BM25 4.58/22.37；单元固定顺序 1.48/12.25（说明单元顺序没有泄露答案）。
- member 2 的 `load_chunks` 能直接读 `webqsp_text/{dev,test}/chunks_slim.parquet`；所有可评题的 gold 段落都在本题文档内。
- `embed_bge.py` + `dense_rank.py` + `score_retrieval.py`：在 QASPER dev 上逐位复现 member 2 公布的 dense 结果（Recall@1/3/5/10 = 33.70/63.00/77.33/94.37，MRR = 0.4213/0.5569/0.5836/0.6006），说明编码配置和指标定义两边一致。
- `qasper_to_gnn.py`：`check_gnn_folder.py` 全部通过；把 `node_feat.npy` 按行求和（= dense 余弦）伪造成 GNN-RAG 的 `.info` 格式，经 `from-gnn` 转换后打分与 dense 完全相同，说明节点 id、实例 key、答案与 chunk 的映射都对。
- QASPER dev 下限：随机（50 次平均）R@1/3/5/10 = 14.31/38.98/58.08/86.36；阅读顺序 6.38/26.20/43.74/79.79；BM25 24.90/56.20/73.79/93.18。

- 2026-11：三个数据集都已完成 ReaRev 训练、测试集推理与 Dense 对比，结果见 [`docs/experiments/gnn_rag/`](../docs/experiments/gnn_rag/README.md)。
