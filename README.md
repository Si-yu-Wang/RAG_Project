# RAG on QASPER & PubMedQA · 项目文档

> 五人协作的检索增强生成（RAG）课程项目。本文件是**项目总入口**。
> 当前状态：**数据与 Chunking（M1）、Dense RAG baseline（M2）已完成**；数据集为 QASPER + PubMedQA。

---

## 1. 项目与分工

| 成员 | 职责 |
|---|---|
| Member 1 | 数据预处理与 Chunking（已完成） |
| Member 2 | Dense RAG baseline（已完成，已合并） |
| Member 3 | Graph |
| Member 4 | Graph |
| Member 5 | Graph |

> Member 3 / 4 / 5 的具体分工待定，暂不细分。

---

## 2. 数据集

两个数据集均已处理成统一 schema（同一套 `paragraphs` / `qa` / `chunks` 三张表），可按需选用：

| 属性 | QASPER v0.3 | PubMedQA-1k |
|---|---|---|
| 论文 / 问题 | 1,169 / 3,598 | 1,000 / 1,000 |
| 段落 | 61,317 | 3,358 |
| 正文 token | 6,433,441 | 307,024 |
| 平均 token / 篇 | 5,504 | 307 |
| 证据标注 | 段落级（2,991 题可用） | 无（仅 yes/no/maybe 判定） |
| 默认切分下 chunks/篇 | 15.10 | 1.02 |

两者的任务形态不同：QASPER 是「给定问题，在论文中检索证据段落」；PubMedQA 是「给定摘要做三分类」，且不带段落级证据标注。下游可依据实验目标选择其一或两者并用。

---

## 3. 阶段一完成内容：数据与 Chunking（M1）

| 阶段 | 产出 |
|---|---|
| 摸底 → 清洗 → 展平 → 证据对齐 | `paragraphs.parquet`（61,317 段）+ `qa.parquet`（3,598 题，对齐率 99.97%） |
| 切分 → chunk 表 → 金标准映射 | 默认配置 `size=512 / overlap=0.2 / soft` → **17,653 chunks**，2,991 题全部可映射 |
| 切分参数实验 | 24 组网格 + 硬/软边界消融（`docs/experiments/chunk_grid_results.md`） |
| 交付契约 | `data_contract.md` + 可执行校验器（24 项检查，53 条单测） |
| PubMedQA 副线 | 转成同一 schema（1,000 题 / 3,358 段 / 1,024 chunks） |

关键数字（QASPER）：索引放大 1.315×；段落被切散比例 3.66%；答案与证据同 chunk 比例 94.54%；`oracle_recall@1 / @5` = 81.62% / 99.97%。

---

## 4. 数据与交付物

以下产物**均已入库，clone 后直接可用**。

### QASPER

| 路径 | 内容 | 行数 |
|---|---|---|
| `data/processed/paragraphs.parquet` | 段落级坐标系 | 61,317 |
| `data/processed/qa.parquet` / `qa_dataset.jsonl` | 评测金标准 | 3,598 |
| `data/processed/chunks/size512_ov102_soft/chunks.parquet` | 默认配置 chunk 表 | 17,653 |
| `data/processed/chunks/size512_ov102_soft/qa_with_gold.jsonl` | 加了 `gold_chunk_ids` | 3,598 |
| `data/processed/chunks/size512_ov102_soft/manifest.json` | 配置哈希 + 校验结果 | — |

### PubMedQA

| 路径 | 内容 | 行数 |
|---|---|---|
| `data/processed/pubmedqa/paragraphs.parquet` | 段落级坐标系 | 3,358 |
| `data/processed/pubmedqa/qa.parquet` / `qa_dataset.jsonl` | 题目（yes/no/maybe） | 1,000 |
| `data/processed/pubmedqa/chunks/pubmedqa_512_ov102/chunks.parquet` | chunk 表 | 1,024 |
| `data/processed/pubmedqa/chunks/pubmedqa_512_ov102/qa_with_gold.jsonl` | chunk 级映射 | 1,000 |
| `data/raw/pubmedqa/ori_pqal.json` | 原始数据（复现用） | — |

其他公共资源：`data/raw/bge_tok/tokenizer.json`（重新跑管道必需）。

> 两种格式（parquet / jsonl）是同一份数据的两个视图，字段与类型一致。字段语义见 [`docs/contract/data_contract.md`](docs/contract/data_contract.md)。

---

## 5. 阶段二完成内容：Dense RAG baseline（M2）

实现位于 [`baselines/member2_dense_rag/`](baselines/member2_dense_rag/README.md)：一套通用 `Baseline` 编排（检索 → 上下文组织 → 生成 → 结果记录），内置 BGE + FAISS 稠密检索；检索器与生成器均按协议可替换，图检索 / 融合检索可接入同一流程。

| 项 | 配置 |
|---|---|
| 检索器 | `BAAI/bge-small-en-v1.5`（revision 固定），L2 归一化 + FAISS `IndexFlatIP`，余弦相似度 |
| 检索范围 | 论文内（`scope=paper`）：先按 `paper_id` 限定候选，再取 Top-K |
| 索引 | dev **3,940** chunks；切分配置 `size512_ov102_soft`，`config_hash=9febc4ee8cb3` |
| 评测集 | QASPER dev，1,005 题；有 gold 的 **890** 题计入分母，排除 115 题（no_evidence 78 / float_only 35 / align_fail 2） |
| 生成器 | `preview`（占位实现，不调用 LLM）；本次只评**检索**，未评答案质量 |

**QASPER dev 检索结果（段落级 macro，分母 890）**

| K | Recall@K | MRR@K |
|---|---:|---:|
| 1 | 33.70% | 0.4213 |
| 3 | 63.00% | 0.5569 |
| 5 | 77.33% | 0.5836 |
| 10 | 94.37% | 0.6006 |

- 对照切分上界（**dev 890 题**、同 `para_id` 口径）：`oracle_recall@1/3/5/10 = 88.42 / 99.76 / 99.99 / 100.00%`。**@10 已吃到上界的 94.4%，瓶颈在排序精度（@1 仅为上界的 38%）。**
- 检索耗时约 **10.1 ms/题**（本机单次，不含建索引与模型加载）。
- 这是**已知论文内检索**，不是跨论文开放检索；dev 为验证集，不能当独立 test 成绩。

> PubMedQA 走同一数据接口，但任务是 yes/no/maybe 三分类、且无段落级 gold（段落 recall 不适用），本次未评测。

**接入注意**：该 baseline 的严格评分入口只接受「论文内 Top-K」结果；图 / 融合检索接入时须按其 Retriever 协议返回 `chunk_id / text / score / rank`，并遵守返回条数的校验（见其 README 的「Retriever」与「完整检索质量评测」两节）。

---

## 6. 下游怎么接

读这些文件的最小依赖：`pip install pandas pyarrow`（重跑管道：`pip install -e ".[dev]"`，Python ≥ 3.11）。

### 检索（Dense / Graph 通用）
读 `chunks.parquet` 的 `chunk_id` + `text` + `n_tokens` 编码、建索引。

> 已有可复用的稠密检索参考实现：`baselines/member2_dense_rag/`（BGE + FAISS，含完整 dev 评测入口）。

### 建图（Graph）
`chunks.parquet` 已含 `section_path`（S2ORC 层级）、`prev_chunk_id` / `next_chunk_id`、`para_ids`，文内结构边无需额外数据。跨文档边需自建：QASPER 的论文之间直接互引实测为 0，只能用文献耦合 / 实体共现 / 语义 kNN（见 [`docs/experiments/citation_graph_probe.md`](docs/experiments/citation_graph_probe.md)）。

### 评测
读 `qa.parquet` + 对应配置的 `qa_with_gold.jsonl`。recall 以段落为单位：

```python
retrieved = topk(...)                       # list[chunk_id]
hit = set()
for cid in retrieved:
    hit |= set(chunks[cid]["para_ids"])     # chunk 展开成段落
recall = len(hit & set(qa[qid]["gold_para_ids"])) / len(gold)
```

QASPER 的 recall 分母取 `zero_gold_reason == ""` 的 **2,991** 题。

---

## 7. 注意事项

### 评测口径

1. **QASPER 分母是 2,991，不是 3,598。** 
2. **recall 以 `para_id` 为单位。** gold 是段落级；`chunk_id` 随配置变，跨 size 用 chunk 级 recall 不可比。
3. **多段证据题占 37.2%**（1,112 / 2,991）；仅 19.5% 的题能在单个 chunk 内看全证据（每题 gold chunk 数 mean 1.76 / max 10）。建议汇报 top-k 而非只 top-1。
4. **`co_split_rate` = 94.54%**：约 5.5% 的题其答案与证据被切到不同 chunk。

### 防泄漏

5. **`qa*` 文件只用于评测**，不得进入索引构建、建图、融合重排。
6. **列名检查挡不住逻辑泄漏**：用「是否含 gold 段落」当特征、或调参后再在同一份 dev 上报指标，机器发现不了。判据：删掉 `qa.parquet` 后建图流程若跑不起来，即是泄漏。

### 字段语义

7. **`para_spans` 的偏移相对「段落自己的文本」**；`char_start` / `char_end` 相对**论文扁平文本**。两者不能混用。
8. **检索用 `text_clean`**，`text_raw` 含 `[REF]` / `[TABLE]` / `[FIGURE]` 占位符。
9. **`len(para_spans) == len(para_ids)`，每段一项。**
10. **`chunk_id` 只在同一配置内稳定**，换 `chunk_size` 后 `c001` 指向不同文本。

### 工程约定

11. **overlap 造成的相邻 chunk 文本重复是设计意图**；「段落被复制」（`paras_dup_pct`）与「段落被切散」（`paras_torn_pct`）是两件事。
12. **没有 test split**：官方 v0.3 包只有 train（888 篇）+ dev（281 篇），test 标注不在包内。
13. **改 schema 三处同改**：`docs/contract/data_contract.md` + `src/rag_data/validate.py` 的 `CHUNK_REQUIRED` + 校验项，再跑一次全量校验。

---

## 8. 复现

> 已入库的数据可直接用，无需重跑。下面命令仅在换参数、重新生成或从零复现时需要。

```powershell
$PY = C:\Users\zxt123\.workbuddy\binaries\python\envs\ragdata\Scripts\python.exe
$env:PYTHONPATH = "src"

# QASPER
$PY scripts/00_profile_dataset.py --data-dir data/raw/qasper --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json
$PY scripts/01_prepare_corpus.py --data-dir data/raw/qasper `
    --processed-dir data/processed --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json
$PY scripts/02_align_evidence.py --data-dir data/raw/qasper `
    --processed-dir data/processed --out-dir docs
$PY scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 --section-boundary soft `
    --name size512_ov102_soft --tokenizer data/raw/bge_tok/tokenizer.json `
    --processed-dir data/processed --out-dir docs
$PY scripts/05_validate_contract.py --processed-dir data/processed `
    --out docs/contract/contract_validation.md

# PubMedQA
$PY scripts/10_prepare_pubmedqa.py --raw-dir data/raw/pubmedqa `
    --processed-dir data/processed/pubmedqa --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json

# 测试
$PY -m pytest tests -q -p no:cacheprovider
```

原始数据来源：QASPER `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`；
PubMedQA `https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`（脚本会自动下载）。

---

## 9. 待确认与文档地图

### 待确认（不影响当前使用，但可能影响 schema 是否要改第二遍）

| # | 事项 | 当前默认 |
|---|---|---|
| Q2 | 图建在 chunk 还是 paragraph 粒度、需要哪些边 | 按 chunk；文内边信息已齐 |
| Q3 | 生成时 top-k 与 token 预算 | 报告 k ∈ {3, 5, 10} |
| Q4 | recall 聚合口径、`unanswerable` 是否进分母 | 分母 2,991、题级 macro、不可回答转拒答指标 |

### 文档地图

| 我想… | 读这份 |
|---|---|
| 接数据 / 查字段 / 确认评测口径 | [`docs/contract/data_contract.md`](docs/contract/data_contract.md) ← **权威口径** |
| 看整体进度、决策记录 | [`docs/PROGRESS.md`](docs/PROGRESS.md) |
| 看切分参数怎么选、召回上界 | [`docs/experiments/chunk_grid_results.md`](docs/experiments/chunk_grid_results.md) |
| 确认数据可用性 | [`docs/contract/contract_validation.md`](docs/contract/contract_validation.md) |
| 做图检索、查跨文档边 | [`docs/experiments/citation_graph_probe.md`](docs/experiments/citation_graph_probe.md) |
| 数据集本身长什么样 | [`docs/dataset/dataset_profile.md`](docs/dataset/dataset_profile.md)、[`docs/dataset/dataset_consistency.md`](docs/dataset/dataset_consistency.md) |
| PubMedQA 的详细说明 | [`docs/dataset/pubmedqa_report.md`](docs/dataset/pubmedqa_report.md) |
| 跑 Dense RAG baseline / 看检索指标口径 | [`baselines/member2_dense_rag/README.md`](baselines/member2_dense_rag/README.md) |
| QASPER dev 检索评测详情 | [`baselines/member2_dense_rag/docs/QASPER_dev_检索评测报告.md`](baselines/member2_dense_rag/docs/QASPER_dev_检索评测报告.md) |
| 和 CG-RAG 论文的数字可比性 | [`docs/design/跨数据集可比性说明.md`](docs/design/跨数据集可比性说明.md) |
