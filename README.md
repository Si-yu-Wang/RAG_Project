# RAG on QASPER · 项目文档

> 五人协作的检索增强生成（RAG）课程项目。本文件是**项目总入口**，由 Member 1 维护。
> 当前状态：**数据与 Chunking 阶段已交付**（tag `data-v1.0`，已合入 `main`）。
> 最后更新：2026-11-01

---

## 1. 项目与分工

| 成员 | 职责 | 主要产物 |
|---|---|---|
| **Member 1** | **数据预处理与 Chunking** | `paragraphs` / `qa` / `chunks` 三张表、`data_contract.md`、参数实验 |
| Member 2 | Dense RAG baseline | BGE 编码 + FAISS 索引 + Top-K 检索 |
| Member 3（Graph） | 论文关系数据 + 图检索 | 边表、1-hop / 2-hop / PPR |
| Member 4 | 融合与重排序 | RRF / Cross-Encoder |
| Member 5 | 统一评测与消融 | 指标实现、实验矩阵、报告 |

**数据集**：QASPER v0.3 —— 1,169 篇 NLP 论文 / 3,598 个问题，带**段落级**证据标注。
主任务是「给定问题，在论文中检索证据段并生成答案」。

---

## 2. Member 1 这一阶段完成了什么

### 2.1 一句话概括

**把 QASPER 的原始 JSON 变成三张下游可直接使用、且被机器校验过的表**，并用模型无关的指标定下了切分参数——顺带量化了「切分到底给下游留了多少空间」。

### 2.2 十个阶段

| # | 阶段 | 结论 / 产物 |
|---|---|---|
| A1 | 工程基建 + Schema 摸底 | `dataset_profile.md`：确认 `paragraphs` 是纯字符串列表、`section_name` 篇内重名率 0% |
| A2 | 清洗（带 offset map） | 无损契约全过；S2ORC 占位符 `BIBREF/TABREF/FIGREF` 共 58,354 处替换 |
| A3 | 展平为段落索引 | `paragraphs.parquet`：**61,317 段 / 6,433,441 tokens** |
| A4 | 证据对齐 | 文本 evidence → 段落坐标，**对齐率 99.97%**（仅 2 条失败） |
| A5 | Chunker | `SentenceAwareTokenChunker`：句级切分 + token 预算 + 段落溯源 |
| A6 | chunk 表导出 + 校验 | 不变量全过：重复 id 0 / 空 chunk 0 / 超预算 0 / 内容缺口 0 |
| A7 | 段级金标准 → chunk 级 | 2,991 题全部映射成功，**0 条不可达** |
| A8 | 切分参数实验 | 24 组网格 + 硬/软边界消融，引入 `oracle_recall@k` 召回上界 |
| A9 | 交付契约 | `data_contract.md` + 可执行校验器（24 项检查 / 53 条单测） |
| B1–B4 | PubMedQA 副线 | 转成同一 schema，验证管道可迁移；**结论是它承载不了 chunk 级实验** |

### 2.3 关键实测数字

| 指标 | 值 |
|---|---|
| 论文 / 问题 / 段落 | 1,169 / 3,598 / 61,317 |
| 正文 token | 6,433,441（mean 5,504 / 篇，max 36,529） |
| **可用评测分母** | **2,991**（不是 3,598，原因见 §5.1） |
| 默认切分配置 | `size=512, overlap=102, section=soft` → **17,653 chunks** |
| 索引放大 | 1.315× |
| 段落被切散比例（`torn`） | 3.66% |
| 答案与证据同在 chunk 比例 | **94.54%** |
| **召回上界 `oracle_recall@1 / @5`** | **81.62% / 99.97%** |
| 单测 | `pytest` **53 passed** |

### 2.4 三个容易被忽略的判断

**① 段落级 provenance 是整个评测的地基。** 每个 chunk 都记录 `para_ids` + `para_spans`，所以 chunk 级的检索结果能还原成段落级证据。没有它，`chunk_id` 会随 `chunk_size` 变化，**跨配置的指标根本不可比**。

**② `oracle_recall@k` 把「检索器弱」和「切分把答案扔了」分开了。** 它假设排序器提前知道答案、只受切分约束。结论是：**k≥5 时切分不是瓶颈**（`or@5` ≥99.2%），瓶颈在排序。这意味着不必在切分参数上继续投入。

**③ overlap 的真实作用是「修复边界撕裂」，不是「增加上下文」。** 它把下一个 chunk 的起点推到段落起始之前，让段落完整落进后一个 chunk：size=512 时 `torn` 从 14.26% 降到 2.00%。

---

## 3. 数据与交付物在哪

> ⚠️ **数据不入库**（体积大，`data/` 整个目录在 `.gitignore` 里）。
> clone 之后必须重跑管道（见 §6），或找 Member 1 要生成好的 `data/processed/`。

| 路径 | 内容 | 行数 |
|---|---|---|
| `data/processed/paragraphs.parquet` | 段落级坐标系（**所有下游的锚点**） | 61,317 |
| `data/processed/qa.parquet` / `.jsonl` | 评测金标准 | 3,598 |
| `data/processed/chunks/size512_ov102_soft/chunks.parquet` / `.jsonl` | **默认配置的 chunk 表** | 17,653 |
| `data/processed/chunks/size512_ov102_soft/qa_with_gold.jsonl` | 加了 `gold_chunk_ids` | 3,598 |
| `data/processed/chunks/<config>/manifest.json` | 配置哈希 + 校验结果 + 统计 | — |
| `docs/data_contract.md` | **字段语义 / 坐标系 / 评测口径 / 防泄漏红线** | — |

**两种格式（parquet / jsonl）是同一份数据的两个视图，字段与类型完全一致。**
parquet 里 `para_spans` 存 JSON 文本（pyarrow 无法推断元组列），jsonl 里是真正的数组。

---

## 4. 下游怎么接

### Member 2（Dense Retriever）
读 `chunks.parquet` 的 `chunk_id` + `text` + `n_tokens`，用 BGE 编码、建 FAISS。
**`n_tokens` 是 BGE tokenizer 口径，最大 511，不会超 512 上限。**

### Member 3（Graph）
`chunks.parquet` 里已有 `section_path`（S2ORC 层级，如 `Experiments > Ablation`）、`prev_chunk_id` / `next_chunk_id`、`para_ids` —— **文内结构边不需要额外数据**。
**但跨文档边要自己解决**：QASPER 的 1,169 篇论文之间**直接互引实测为 0**（Semantic Scholar + OpenAlex 双源抽样验证），只能用文献耦合 / 实体共现 / 语义 kNN。详见 [`docs/citation_graph_probe.md`](docs/citation_graph_probe.md)。

### Member 4（融合重排）
`chunk_id` 作唯一主键即可。各路召回结果统一按 `chunk_id` 合并，不要自己拼 id。

### Member 5（评测）
读 `qa.parquet` + 本配置的 `qa_with_gold.jsonl`。**recall 的正确算法只有一种**：

```python
retrieved = topk(...)                       # list[chunk_id]
hit = set()
for cid in retrieved:
    hit |= set(chunks[cid]["para_ids"])     # chunk 展开成段落
recall = len(hit & set(qa[qid]["gold_para_ids"])) / len(gold)
```

分母取 `zero_gold_reason == ""` 的 **2,991** 题。

---

## 5. 接手前必读：注意事项

> 每条都写清了「踩了会怎样」，按影响排序。

### 5.1 评测口径（最容易错，错了指标全废）

1. **分母是 2,991，不是 3,598。** 607 道题没有 gold 证据段：331 道不可回答、100 道可回答但标注者未选证据、174 道证据只有图表、2 道对齐失败。**把这 607 题当 0 分会系统性拉低 recall，而且看起来像是检索出了问题。**
2. **recall 一律以 `para_id` 为单位。** gold 是段落级的；`chunk_id` 随配置变，用 chunk 级 recall 跨 size 比较是错的。
3. **top-1 在 QASPER 上没有意义。** 只有 **19.5%** 的题能把全部证据装进 1 个 chunk（每题 gold chunk 数 mean 1.76 / max 10）。汇报必须给 top-k。
4. **`co_split_rate` = 94.54%**，即约 5.5% 的题**即使检索完美也必然答错**——证据和答案被切到了不同 chunk。报端到端指标时要单独说明，否则会被误读成系统缺陷。

### 5.2 防泄漏（错了看不出来）

5. **`qa*` 四份文件只能给 Member 5。** 不得进入 M2 的索引构建、M3 的建图、M4 的融合重排。
6. **列名检查挡不住逻辑泄漏。** 校验器会拒收含 `gold` / `answer` / `evidence` / `extractive` 的 public 列，但「用是否含 gold 段落当特征」、或「用标注调完参再在同一份 dev 上报指标」，机器发现不了。**自查判据：删掉 `qa.parquet` 后建图流程如果跑不起来，就是泄漏了。**

### 5.3 字段语义（错了不报错，只是结果错）

7. **`para_spans` 的偏移相对「段落自己的文本」**，不是相对 chunk。`char_start/char_end` 才是相对的**论文扁平文本**坐标。两者不能混用。
8. **`text_raw` 含占位符**（`[REF]` / `[TABLE]` / `[FIGURE]`），检索一律用 `text_clean`。
9. **`para_spans` 一段一项**（`len(para_spans) == len(para_ids)`）。早期版本按句子写入导致重复，已修，别按旧假设写代码。
10. **`chunk_id` 只在同一配置内稳定**，换 `chunk_size` 后 `c001` 指向的文本完全不同。

### 5.4 工程约定（错了会返工）

11. **overlap 造成的「相邻 chunk 文本重复」是设计意图，不是 bug。** 段落被复制到两个 chunk（`paras_dup_pct`）和段落被切散（`paras_torn_pct`）是两件事，混为一谈会把 overlap 误判成有害。
12. **没有 test split。** 官方 v0.3 包只有 train（888 篇）+ dev（281 篇），test（416 篇）的标注不在包里。评测只能用 dev，或从 train 再切一份 holdout。
13. **改 schema 必须三处同改**：`docs/data_contract.md` + `src/rag_data/validate.py` 的 `CHUNK_REQUIRED` + 校验项，然后重跑一次全量校验。**不允许只改代码不改契约。**

### 5.5 结论的边界（别过度推广）

14. **`size=512 / overlap=0.2 / soft` 是数据层最优，不是最终结论。** 它的依据是 `oracle_recall`（与 embedding 无关的上界），**硬/软 section 边界的取舍必须由 Member 2 的真实 recall 复核**：硬边界给出的 chunk 更短但更完整（`torn` 2.00% vs 3.66%），软边界给出的更长但更容易切散。
15. **PubMedQA 只能当冒烟测试，不能当第二主实验。** 它平均 307 token/篇，默认 512 切分下 **976/1,000 篇被切成恰好 1 个 chunk**，chunk 网格与图检索都无从做起。详见 [`docs/pubmedqa_report.md`](docs/pubmedqa_report.md)。

---

## 6. 复现

```powershell
$PY = C:\Users\zxt123\.workbuddy\binaries\python\envs\ragdata\Scripts\python.exe
$env:PYTHONPATH = "src"

# 0) 下载 BGE tokenizer 为单文件（绕开 HF 缓存）
$PY scripts/dev/fetch_tokenizer.py --out data/raw/bge_tok/tokenizer.json

# 1) 摸底
$PY scripts/00_profile_dataset.py --data-dir data/raw/qasper --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json

# 2) 清洗 + 展平 → paragraphs.parquet
$PY scripts/01_prepare_corpus.py --data-dir data/raw/qasper `
    --processed-dir data/processed --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json

# 3) 证据对齐 → qa.parquet（对齐率应为 99.97%）
$PY scripts/02_align_evidence.py --data-dir data/raw/qasper `
    --processed-dir data/processed --out-dir docs

# 4) 切分 → chunks（应得到 17,653 条）
$PY scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 --section-boundary soft `
    --name size512_ov102_soft --tokenizer data/raw/bge_tok/tokenizer.json `
    --processed-dir data/processed --out-dir docs

# 5) 契约校验（应为 24 PASS / 0 FAIL）
$PY scripts/05_validate_contract.py --processed-dir data/processed `
    --chunks-dir data/processed/chunks/size512_ov102_soft `
    --out docs/contract_validation.md

# 6) 单测（应为 53 passed）
$PY -m pytest tests -q -p no:cacheprovider
```

原始数据 `qasper-train-dev-v0.3.tgz`（10.8 MB）：
`https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`
解压后放到 `data/raw/qasper/`。

---

## 7. 还缺三个外部输入

这三项**不影响现在使用**（契约里都给了默认值），但会影响 schema 是否需要改第二遍：

| # | 待确认 | 问谁 | 当前默认 |
|---|---|---|---|
| Q2 | 图建在 chunk 还是 paragraph 粒度、还需要哪些边 | Member 3 | 按 chunk；`para_ids` / `section_path` / `prev`-`next` 已够建文内边 |
| Q3 | 生成时 top-k 几、token 预算多少 | Member 4 | 报告 k ∈ {3, 5, 10} |
| Q4 | recall 聚合口径、`unanswerable` 是否进分母 | Member 5 | 已定：分母 2,991、题级 macro、`unanswerable` 转拒答指标 |

---

## 8. 文档地图

| 我想… | 读这份 |
|---|---|
| 接数据 / 查字段 / 确认评测口径 | [`docs/data_contract.md`](docs/data_contract.md) ← **权威口径** |
| 看整体进度、决策记录、全部已知坑 | [`docs/PROGRESS.md`](docs/PROGRESS.md) |
| 看切分参数怎么选的、召回上界 | [`docs/chunk_grid_results.md`](docs/chunk_grid_results.md) |
| 确认这份数据能不能用 | [`docs/contract_validation.md`](docs/contract_validation.md) |
| 做图检索、想知道有没有跨文档边 | [`docs/citation_graph_probe.md`](docs/citation_graph_probe.md) |
| 数据集本身长什么样 / 与论文是否一致 | [`docs/dataset_profile.md`](docs/dataset_profile.md)、[`docs/dataset_consistency.md`](docs/dataset_consistency.md) |
| 用 PubMedQA 做第二数据集 | [`docs/pubmedqa_report.md`](docs/pubmedqa_report.md) |
| 和 CG-RAG 论文的数字能不能比 | [`docs/跨数据集可比性说明.md`](docs/跨数据集可比性说明.md) |
