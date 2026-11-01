# 项目进度 · Member 1（Dataset & Chunking）

> 分支：`feature/dataset-chunking`　|　最后更新：2026-11-01
> 本文件只记录**已完成 / 待办 / 关键决策**；技术细节见 [`Dataset_Chunking_技术清单.md`](Dataset_Chunking_技术清单.md)，实测数据见 [`dataset_profile.md`](dataset_profile.md)、[`cleaning_report.md`](cleaning_report.md)。

---

## 一、进度总览

> 计划已于 2026-11-01 改为 **v2（双数据集 + 建图移交 M3 + 证据对齐提前）**，详见 [`M1_实施计划_v2.md`](M1_实施计划_v2.md)。下表编号与 v2 对齐，括号内为旧编号。

| # | 任务 | 状态 | 交付物 |
|---|---|---|---|
| A1 (P0/P1) | 工程基建 + Schema 摸底 | ✅ 完成 | `pyproject.toml`、`.gitignore`、`tests/`、`docs/dataset_profile.md`、`docs/figures/*.png` |
| A2 (P2) | 清洗（带 offset map） | ✅ 完成 | `src/rag_data/cleaner.py`、`docs/cleaning_report.md` |
| A3 (P3) | 展平为段落索引 | ✅ 完成 | `src/rag_data/flattener.py`、`data/processed/paragraphs.parquet` |
| **A4 (P6)** | 证据对齐 → `qa.parquet` / `qa_dataset.jsonl` | ✅ **完成（对齐率 99.97%）** | `docs/evidence_alignment_report.md` |
| A5 (P4) | Chunker 实现 | ✅ **完成（13 条单测）** | `src/rag_data/chunker.py` |
| A6 (P5) | chunk 表导出 + 校验 | ✅ **完成** | `data/processed/chunks/<config>/chunks.{parquet,jsonl}` |
| A7 | 段级金标准 → chunk 级 | ✅ **完成**（`gold_chunk_ids`，0 条不可达） | `qa_with_gold.jsonl`、`docs/chunking_stats_*.md` |
| A8 (P7) | 切分实验（24 组网格 + 硬/软边界消融） | ✅ **完成** | `docs/chunk_grid_results.md`、`docs/chunk_grid_hard.md`、`docs/grid_results*.csv` |
| **A9 (P8)** | **交付：契约 + 校验器 + 防泄漏 + tag** | ✅ **完成（24 项校验全过）** | `docs/data_contract.md`、`src/rag_data/validate.py`、`scripts/05_validate_contract.py`、`docs/contract_validation.md` |
| B1–B4 | PubMedQA-1k 副线（冒烟测试 + 泛化性） | ✅ **完成**（管道通用，但切分退化） | `docs/pubmedqa_report.md`、`scripts/10_prepare_pubmedqa.py`、`docs/contract_validation_pubmedqa.md` |
| C1–C3 | 契约文档 / 通用校验器 / 防泄漏校验 | ✅ **并入 A9 完成** | 同上 |

**已移交给其他成员**（原 N1–N2 属 M1，现归 M3）：参考文献抓取、耦合/共被引计算、实体抽取、图构建、图查询。

**测试**：`pytest` **53 passed**（loader / cleaner / profiling / evidence_aligner / chunker / oracle / validate）。

### A9 交付（契约 + 可执行校验）

| 产物 | 内容 |
|---|---|
| `docs/data_contract.md` | **下游唯一权威口径**：consumer matrix、23 个字段逐条语义、三段坐标系、评测分母 2,991 的推导、`co_split_rate`、防泄漏红线、9 条反模式 |
| `src/rag_data/validate.py` | 可执行契约：18 项 chunks 检查 + 6 项 qa 检查，纯 Python、可单测 |
| `scripts/05_validate_contract.py` | 对任一配置跑全量校验，输出 `docs/contract_validation*.md` |
| `docs/contract_validation.md` | **24 PASS / 0 WARN / 0 FAIL**（soft 与 hard 两个配置都是） |
| `tests/test_validate.py` | 8 条，把"泄漏必须报错""NaN≠None""重复 span 必须拒收"等钉死 |

**校验器一上线就抓到 3 个真实缺陷**（都已修）：

1. **`para_spans` 按句子重复写入** —— 一个 4 句段落输出 4 条相同记录，`len(para_spans) != len(para_ids)`。指标侧因为用 dict 去重才没出错，但按 `zip` 读的下游会错位。
2. **jsonl 里 `gold_para_ids` 是 ndarray 的 repr 字符串** —— `json.dumps(default=str)` 把 ndarray 序列化成了 `"['1503...#p0019' ...]"`，是合法 JSON 但类型错误。M5 若读 jsonl 会拿到字符串而非列表。
3. **`prev_chunk_id` 被写成 `NaN`** —— 严格 JSON 解析器直接拒绝；jsonl 里 `para_spans` 也还是字符串而非数组。

修法：`03_build_chunks.py` 加 `jsonable()` 递归转换 + `allow_nan=False` 写出，两个格式的字段类型现在完全一致。

### B 线（PubMedQA-1k）实测：管道通用，但切分退化

完整报告见 [`pubmedqa_report.md`](pubmedqa_report.md)。核心是三个数字：

| 配置 | chunks | **chunks/篇** | 单段 chunk 占比 |
|---|---:|---:|---:|
| **512 / 0.2**（项目默认） | 1,024 | **1.02** | 2.0% |
| 128 / 0.2 | 4,230 | 4.23 | 51.4% |
| *（对照）QASPER 512 / 0.2* | *17,653* | ***15.10*** | *—* |

- PubMedQA 平均 **307 token/篇**、**max 648**，3.36 段/篇 → 默认 512 下 **976/1,000 篇被切成恰好 1 个 chunk**。**切分在这里是空操作**，chunk 网格与图检索都无从做起。
- 与 QASPER 相差 **18×**（token/篇 307 vs 5,504），这是数据形态差异，不是参数问题。
- **清洗规则完全空转**：QASPER 的 S2ORC 占位符规则命中 58,354 次，在 PubMedQA 上 **0 次**（后者是已清洗的 PubMed 摘要）。**管道代码通用，清洗配置不通用。**
- **真正有价值的产出**：同一套 `chunker` + `03_build_chunks` + `05_validate_contract` 不改一行代码跑通，契约校验 **23 PASS / 1 WARN / 0 FAIL**（唯一 WARN 是自加的 `decision` 列）。**证明 schema 可迁移**。
- **建议用法**：只做「文档级检索冒烟测试」（1,000 候选找对摘要，恰好补上 QASPER 缺的"找对论文"这一环）+ 端到端 yes/no/maybe 准确率。**明确不做** chunk 网格、图检索、段落级 recall、与 QASPER 并排比表。

### A5–A7 实测（size=512, overlap=102, min=64）

| 配置 | chunks | 平均长度 | 索引放大 | `co_split_rate` | 跨 section chunk |
|---|---|---|---|---|---|
| **soft**（默认） | **17,653** | **479** | 1.315× | **94.54%** | 12,776（72%） |
| hard | 23,851 | 304 | 1.126× | 93.36% | 0 |

- 不变量全部通过：重复 id 0 / 空 chunk 0 / 超预算 0 / **内容级缺口 0**（4 处缺口均为空白字符，落在 chunk 边界上）/ 邻居链断裂 0
- **gold 可达性**：2,991 题有 gold 段落，映射到 chunk 后 **0 条不可达**

> ⚠️ **订正（2026-11-01，A8 之后）**：此处原先写的"**hard section 边界是净损失**"是错的。
> 当时只用 `co_split_rate` 与 chunk 数判断，漏掉了「段落是否被切散」这个更关键的维度。
> A8 用 `oracle_recall@1` 与 `paras_torn_pct` 重测后，hard 在**证据完整性上反而更好**
> （512/0.2：torn 3.66% → 2.00%，or@1 81.62% → 85.26%），代价是 chunk 平均只有 304 token、数量多 35%。
> 详见下方「A8 实测」。

### A8 实测（24 组网格，size 128–768 × overlap 0–0.3）

完整表见 [`chunk_grid_results.md`](chunk_grid_results.md)。核心是引入了**模型无关的召回上界**：

> `oracle_recall@k` = 假设排序器提前知道答案、只受切分约束时，k 个 chunk 内能看全的 gold 证据比例。
> **任何 Dense / Graph / 融合方法的 recall@k 都不可能超过它。**

| size | ov=0 | ov=0.1 | ov=0.2 | ov=0.3 |
|---|---|---|---|---|
| 128 | 23.89 | 29.25 | 31.25 | 33.41 |
| 256 | 48.63 | 55.24 | 60.94 | 65.49 |
| **512** | 69.95 | 77.93 | **81.62** | 85.33 |
| 768 | 78.12 | 85.51 | 89.16 | 90.46 |

（上表为 `oracle_recall@1`，单位 %）

三条结论：

1. **k≥5 时切分不是瓶颈。** `oracle_recall@5` 最小值 94.05%，size≥256 的配置全部 ≥99.25%。瓶颈在排序，不在切分。**k=1 才会暴露切分代价**（23.89% → 90.46%），所以 M5 不能只报 top-1。
2. **overlap 的真实作用是「修复边界撕裂」，不是「增加上下文」。** 它把下一个 chunk 的起点推到段落起始之前，让段落完整落进后一个 chunk：size=512 时 torn 从 14.26% 降到 2.00%，代价是索引放大到 1.519×。
3. **section 硬边界与软边界各有胜场**（不是单调优劣）：hard 的 chunk 平均只有 soft 的一半长（304 vs 479），chunk 数多 35%，但 `torn` 与 `oracle_recall@1` 更好、总 token 成本更低。**这一项必须等 M2 的真实 recall 才能定。**

**推荐**：主推 `size=512 / overlap=0.2`（即已产出的 `size512_ov102_soft`），满足 BGE 512 上限、`index_blowup` 1.315、`oracle_recall@1` 81.62%、`torn` 3.66%。若不设 512 上限，数据层最优是 `768/0.2`（or@1 89.16%，torn 1.11%）。

**代码**：上界算法已抽成独立模块 `src/rag_data/oracle.py`（`evidence_recall_ceiling` / `paragraph_status`），配 `tests/test_oracle.py` **8 条**单测把语义钉死——因为这个语义被写错过两次（详见下方坑 6）。它不依赖 pandas，M5 想复用可以直接 import。

**一个必须记住的坑**：第一版指标把「段落被 overlap 复制到两个 chunk」当成「段落被撕裂」，得出**"overlap 全面有害"**的错误结论；分开 `torn` / `dup` 后结论反转。这两个概念必须永远分开。

---

## 二、已产出的数据文件

> 入库策略（2026-11-01 修订）：默认忽略 `data/`，**仅放行下游开箱即用约 59 MB 的最小交付集**。
> clone 后 M2/M3/M4/M5 无需重跑管道即可直接开工。

| 文件 | 内容 | 规模 | 是否入库 |
|---|---|---|---|
| `data/processed/paragraphs.parquet` | **段落级坐标系**：61,317 行 | 32 MB | ✅ 入库 |
| `data/processed/qa.parquet` | **评测金标准**：3,598 题，含 `gold_para_ids` | 0.75 MB | ✅ 入库 |
| `data/processed/qa_dataset.jsonl` | 同上（jsonl，给 M5 直接读） | 3.4 MB | ✅ 入库 |
| `data/processed/manifest.json` + `manifest_qa.json` | 清洗/对齐配置 + 校验 + 统计 | 3 KB | ✅ 入库 |
| `data/processed/chunks/size512_ov102_soft/chunks.parquet` | **默认配置 chunk 表** | 18 MB | ✅ 入库 |
| `data/processed/chunks/size512_ov102_soft/qa_with_gold.jsonl` + `manifest.json` | 加了 `gold_chunk_ids` | 3.7 MB | ✅ 入库 |
| `data/raw/bge_tok/tokenizer.json` | BGE tokenizer（单文件，重跑管道必需） | 711 KB | ✅ 入库 |
| `data/raw/qasper/qasper-{train,dev}-v0.3.json` + tgz | 官方原始数据（S3 可下载） | 43 MB + 11 MB | ❌ 需下载 |
| `data/processed/chunks/size512_ov102_hard/` | 硬边界消融对照 | 23,851 chunk | ❌ 需重跑 |
| `data/processed/chunks/*/chunks.jsonl` | jsonl 副本（从 parquet 一行生成） | 50 MB/份 | ❌ 需重跑 |
| `data/processed/chunks/grid_results.csv` / `.json` | A8 网格结果 | — | ❌ 副本在 `docs/` |
| `data/raw/pubmedqa/ori_pqal.json` + `data/processed/pubmedqa/` | PubMedQA 副线 | 2.58 MB | ❌ 需重跑 |
| `docs/grid_results.csv` / `docs/grid_results_hard.csv` | A8 网格结果的入库副本 | 4 KB | ✅ 入库 |

`paragraphs.parquet` 主要字段：`para_id` / `paper_id` / `split` / `section_idx` / `section_name` / `para_idx` / `is_abstract` / `text_raw` / `text_clean` / `char_start` / `char_end` / `n_chars` / `n_tokens` / `offset_map`。

---

## 三、复现命令

```bash
PY=C:\Users\zxt123\.workbuddy\binaries\python\envs\ragdata\Scripts\python.exe

# 0) 一次性：把 BGE tokenizer 下载成单文件（绕开 HF 缓存）
$PY scripts/dev/fetch_tokenizer.py --out data/raw/bge_tok/tokenizer.json

# 1) P1 摸底（加 --no-figures 可跳过 PNG）
$PY scripts/00_profile_dataset.py --data-dir data/raw/qasper --out-dir docs \
    --tokenizer data/raw/bge_tok/tokenizer.json

# 2) P2+P3 清洗与展平
$PY scripts/01_prepare_corpus.py --data-dir data/raw/qasper \
    --processed-dir data/processed --out-dir docs \
    --tokenizer data/raw/bge_tok/tokenizer.json

# 3) A4 证据对齐
$PY scripts/02_align_evidence.py --data-dir data/raw/qasper \
    --processed-dir data/processed --out-dir docs

# 4) A5–A7 构建某一配置的 chunk 表（默认 512/0.2/soft）
$PY scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 \
    --section-boundary soft --name size512_ov102_soft \
    --tokenizer data/raw/bge_tok/tokenizer.json \
    --processed-dir data/processed --out-dir docs

# 5) A8 网格实验（24 组，约 7.5 分钟）
$PY scripts/04_chunk_grid.py --sizes 128,192,256,384,512,768 \
    --overlaps 0,0.1,0.2,0.3 --section-boundary soft \
    --tokenizer data/raw/bge_tok/tokenizer.json \
    --processed-dir data/processed --out-dir docs

# 5b) 硬边界消融（6 组）
$PY scripts/04_chunk_grid.py --sizes 256,384,512 --overlaps 0,0.2 \
    --section-boundary hard \
    --tokenizer data/raw/bge_tok/tokenizer.json \
    --processed-dir data/processed --out-dir docs

# 5c) 只重画报告，不重跑网格（改指标口径时用）
$PY scripts/04_chunk_grid.py --report-only --processed-dir data/processed --out-dir docs

# 6) A9 契约校验（结构 + 覆盖 + gold 可达 + 防泄漏）
$PY scripts/05_validate_contract.py --processed-dir data/processed \
    --chunks-dir data/processed/chunks/size512_ov102_soft \
    --out docs/contract_validation.md

# 7) B 线：PubMedQA-1k（下载 + 转 schema + 同一套切分与校验）
$PY scripts/10_prepare_pubmedqa.py --raw-dir data/raw/pubmedqa \
    --processed-dir data/processed/pubmedqa --out-dir docs \
    --tokenizer data/raw/bge_tok/tokenizer.json
$PY scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 --section-boundary soft \
    --name pubmedqa_512_ov102 --processed-dir data/processed/pubmedqa --out-dir docs \
    --tokenizer data/raw/bge_tok/tokenizer.json

# 测试
$env:PYTHONPATH = "src"; $PY -m pytest tests -q -p no:cacheprovider
```

---

## 四、关键决策记录

| # | 决策 | 依据 |
|---|---|---|
| D1 | 数据源用**官方 JSON**，不用 HF `datasets` 镜像 | 镜像会把 `full_text` 压平，丢失段落结构 |
| D2 | token 计数用 **BGE tokenizer（XLM-R）**，不用 tiktoken | 决定 chunk 尺寸必须与下游 embedding 的 512 上限同口径 |
| D3 | tokenizer 走**本地单文件** | HF 缓存会写 `~/.cache` 并清理 `.incomplete`，在沙箱里会被删除守卫拦死 |
| D4 | 引文占位符替换为 `[REF]/[TABLE]/[FIGURE]` 而**不是删除** | 删除会破坏句子语法（"`BIBREF0` constructed a network" 会变成 "constructed a network"） |
| D5 | 清理保留**逐段 offset map** | 下游若要回原文引用，必须能从 cleaned 偏移反查 |
| D6 | 空段落（1,133 个，1.85%）在展平时丢弃 | 对检索无贡献，且会产生空 chunk |
| D7 | 摘要提升为 `section_idx = -1` 的合成 section | QASPER 的 `abstract` 是顶层字段，不在 `full_text` 里 |
| D8 | **CG-RAG 只是参考文献，不是本项目的基线** | 分工表指定 QASPER；CG-RAG 的实验数据是 PubMedQA-1k + PapersWithCodeQA，全文 0 次提及 QASPER（已用脚本核验），见 `docs/dataset_consistency.md` |
| D9 | **不换数据集，继续用 QASPER；M3 用"文献耦合边"替代"引用边"** | 见下方「六之三」。核心理由：论文的两个图都是作者自建且未公开，换数据集并不自动获得可比性；而实测证明 QASPER 上直接引用边为 0、耦合边可用（25.7%）。详见 `docs/citation_graph_probe.md` |
| D10 | **默认 chunk 配置保持 `size=512 / overlap=0.2 / soft`** | A8 的 24 组网格复核：在 `chunk_size ≤ 512`（BGE 上限）与 `index_blowup ≤ 1.5` 约束下，该配置的 `oracle_recall@5` = 99.97% 为最优。不设 512 上限时的数据层最优是 `768/0.2`（or@1 89.16%），但不适用于 bge-large |
| D11 | **`paras_torn`（真实撕裂）与 `paras_dup`（overlap 复制）必须分开统计** | 混为一谈会得出"overlap 全面有害"的错误结论；分开后结论反转。见 `docs/chunk_grid_results.md` §1 |
| D12 | **A8 的结论只用于**筛选配置范围**，不作为最终定论** | `oracle_recall@k` 与 embedding 无关，只能给出上界。硬/软边界、512 vs 768 这类取舍取决于 BGE 在 300 vs 480 token 上的信噪比，必须由 M2 的真实 recall 复核 |
| D13 | **`para_spans` 一段一项**（`len(para_spans) == len(para_ids)`） | 原先按句子写入导致重复；下游若 `zip(para_ids, para_spans)` 会错位。已由 `spans.paired` / `spans.unique` 两项校验强制 |
| D14 | **评测 recall 一律以 `para_id` 为单位，分母 2,991** | gold 是段落级、chunk 粒度随配置变；607 道无 gold 的题（含 331 道不可回答）必须分流，不能当 0 分。详见 `data_contract.md` §7 |
| D15 | **`data/processed/` 全目录 gitignore，需要入库的产物放 `docs/`** | `grid_results*.csv` 已复制到 `docs/`；`.workbuddy/` 也加入 gitignore（本地会话数据）。**已被 D16 修订** |
| D16 | **数据改为"默认忽略、仅放行开箱即用最小交付集"（约 59 MB）** | 用户指出"clone 后要能直接开工"。放行 `paragraphs.parquet` / `qa.*` / 默认配置 `chunks.parquet` + `qa_with_gold.jsonl` / manifests / `tokenizer.json`；原始 JSON、硬边界消融、jsonl 副本、PubMedQA 保持忽略（可重新生成） |

---

## 五、P2/P3 实测结果摘要

- 段落 **61,317** 条，tokens **6,433,441**；清洗改动过 **44.78%** 的段落
- 规则命中：`refs` 58,354 / `strip` 5,772 / `newline_join` 2,134 / `space_collapse` 1,310 / `nfkc` 140 / `zero_width` 11
- **无损契约全部通过**：重复 `para_id` 0、切片不一致 0/200、offset map 覆盖不一致 0/2000、NFKC 逐字符等价违反 0/3000
- ⚠️ **LaTeX 密集段落 8,896 条（14.51%），占 19.48% 的 token** —— 接近 1/5 的 token 预算花在行间公式上。**是否在切分前剔除需拍板**（见六）

---

## 六、待决策 / 待他人回答

| # | 事项 | 问谁 | 不解决的后果 |
|---|---|---|---|
| Q1 | embedding 是否是 `bge-large-en-v1.5`（512 token 上限） | M2 | **已不阻塞**：A8 按 512 跑了全网格，并同时给出不设上限时的最优（`768/0.2`）。若 M2 确定用长上下文模型，改推后者即可 |
| Q2 | 图建在 chunk 还是 paragraph 粒度、还需要什么边 | M3 | 可能要额外导出边表 |
| Q3 | 生成时 top-k / token 预算 | M4 | 无法定实验里 `oracle_recall@k` 的 k |
| Q4 | recall 按 chunk 还是 para 算；`unanswerable`（占 10%）进不进分母；多答案如何聚合（7.8% 的问题标注者类型不一致） | M5 | `qa.parquet` 字段语义定不下来 |
| Q5 | **是否剔除 14.51% 的 LaTeX 密集段落** | M2 + M5 | 影响信噪比与公式型问题的可检索性 |
| Q6 | 是否同意 dev 调参、test 只在最后跑一次 | 全员 | 全组过拟合评测集 |
| Q7 | **检索任务是"篇内"还是"开集"**：QASPER 原设定是给定论文找证据段，若照做，M2 的 FAISS 全库索引意义会被削弱 | 全员（M2/M3 为主） | 决定 M1 是否需要按"跨论文可检索"重新设计 chunk 元数据 |

---

## 六之二、数据集选型说明（为什么是 QASPER）

**先说清楚：这个选择不是 M1 做的，是分工表里给定的。** 分工表 Member 1 栏原文即 "QASPER 预处理、Chunk ID、Metadata"，M1 是照此执行，并未参与选型权衡。以下是补做的、本应在开工前就给出的评估。

### 支持 QASPER 的理由

| # | 理由 |
|---|---|
| 1 | **PubMedQA-1k 会让切分工作失去意义**：1,000 条「摘要 + yes/no/maybe 标签」，摘要约 200 词，无需切分，P4/P7 无从做起 |
| 2 | **有段落级 gold evidence**，M5 能直接算 evidence recall；PubMedQA 只有标签，算不出"检索有没有找对地方" |
| 3 | **是标准基准**（NAACL 2021），有公开基线可引；PapersWithCodeQA 是 CG-RAG 作者自建，无第三方基线、无公认口径 |
| 4 | **获取成本低**：10.8 MB 单个 tgz；PapersWithCodeQA 需要抓 84 个 leaderboard + 抽参考文献建引文图，本身就是一大块工作量 |

### 代价（必须在组内对齐）

1. **没有引文图**。QASPER 的 1,169 篇论文之间无引用关系，M3 只能建**文档内图**（section 层级 / 段落邻接 / 实体共现）。分工表里"1-hop/2-hop/PPR"的措辞接近 CG-RAG 的引文图设定，**这点必须提前告知 M3**，否则他会找不到跨文档边。
2. **每题绑定单篇论文**。原始设定是"给定论文找证据段"，照做会让 M2 的全库检索退化。需回答 Q7。

### 什么情况下应该换数据集

如果组里的目标其实是**复现 CG-RAG 那条路线**（引文图 + 稀疏/稠密纠缠融合），那 QASPER 是错的选择，应改 PubMedQA-1k + PapersWithCodeQA。**切换窗口只到 P4 之前**：一旦开始切分，边表设计就要重做。已完成的 P1–P3 代码可复用（`paragraphs.parquet` / `chunks.parquet` 的结构与数据集无关）。

---

## 六之三、数据集二次评估结论（2026-11-01，已定）

### 决策：不换数据集，继续用 QASPER v0.3

原提问：既然 M3 是核心工作、又希望和论文数字对比，是否该换到 PubMedQA-1k + PapersWithCodeQA？

**不换。** 三条理由，按决定性排序：

**① 换数据集并不能换来"可比性"——论文的图是作者自己造的，而且没公开。**

- PubMedQA-1k 原始数据集**没有引文关系**。论文 §5.1 原话："The original dataset, however, lacks citation information between papers, which we addressed by extracting the references for each paper and constructing a citation graph database with a total of 7,849 papers."
- PapersWithCodeQA 是作者自建：84 个 leaderboard × 每榜 top-20 论文 → 12,171 篇 arXiv 论文，LaTeX 全文 + 参考文献。
- 检索 GitHub（`CG-RAG` / `LeSeGR`）：**0 个仓库**，论文未提供 code / data 链接。

→ 换过去拿到的只是"没有图的 PubMedQA-1k"和"需要自己重建的 PapersWithCodeQA"。**M3 的图还是得自己造，而且造出来的不等于论文那份。**

**② M3 是否成立，取决于"有没有图"，不取决于"用哪个数据集"。**

分工表里 M3 写的是"**实体关系抽取**、图构建、图查询（1-hop/2-hop/PPR）"，不是"引文图"。在 QASPER 上，跨文档实体共现图天然存在（1,169 篇都是 NLP 论文，共享 BERT / Transformer / GLUE / SQuAD 等实体），文档内层级图也已经由 P3 产出。

**③ 换成 PubMedQA-1k 会让 M1 的工作归零。**

1,000 条「约 200 词摘要 + yes/no/maybe 标签」，摘要无需切分，P4 的 chunker 与 P7 的 size/overlap 实验都失去对象；且它只有标签，没有段落级证据，M5 算不出 evidence recall。

### 对比口径：用相对增益，不用绝对值

CG-RAG 真正可迁移的结论不是 0.961 这个数，而是**图检索相对纯稠密检索的增益**（Hit@1 0.913 → 0.961；PapersWithCodeQA Acc 0.769 → 0.835）。绝对值受 embedding、LLM、chunking 影响太大，跨项目不可比。

本项目报告三档即可：

| 档 | 配置 | 目的 |
|---|---|---|
| B0 | dense-only（M2） | 基线 |
| B1 | dense + 文档内结构图 | 证明结构有用 |
| B2 | dense + 跨文档图（耦合/实体） | 证明跨文档有用，**这一档才是和 CG-RAG 对应的对照** |

### 若确实要复现论文数字

只有一种情况：**课程/导师明确要求复现 CG-RAG**。那必须换，且要接受"重建 12,171 篇论文的图 + 924 道题"的工作量——那已经不是一个数据集切换，而是换项目。

### 面向「CG-RAG 的思路」逐项对比：QASPER vs PubMedQA-1k

前提：建图由 M3 负责，此处只比"哪个数据集更承载得住 CG-RAG 的方法"。

CG-RAG 有三件事必须落地：**① chunk 级上下文图**（文内叙事边 + 跨文档引用边）、**② LeSeGR 稀疏/稠密纠缠的消息传递**、**③ 子图摘要生成**。

| 维度 | QASPER | PubMedQA-1k | 胜者 |
|---|---|---|---|
| 能否切成有意义的 chunk | 每篇 mean 5,364 / max 36,529 token，13.9 个 section → 10–70 个 chunk | 每条是一条 ~200–300 词的摘要 → **1–2 个 chunk**，"chunk 级图"退化成"论文级图" | **QASPER** |
| 文内结构边（component ①） | section 层级 + 段落邻接，61,317 段 | 摘要内部几乎没有结构 | **QASPER** |
| 跨文档边（component ①） | 需自建；**实测互引 0**，只能用耦合/实体 | 需自建（论文作者建了 7,849 篇）；原始数据集无引用 | 打平（都要自建） |
| 多跳/结构是否真的必要 | 55.5% 的答案需要多段落证据 → 结构确实携带信息 | 答案在单条摘要内，**图对答案没有直接贡献** | **QASPER** |
| 段落级评测金标准 | ✅ 完整段落级 evidence | ⚠️ 只有 yes/no/maybe + 少量证据句 | **QASPER** |
| 与论文数字可比 | ❌ 论文没用过 | ✅ 论文在上述报过数（Hit@1 0.961 等） | **PubMedQA-1k** |

**结论：做 CG-RAG 的思路，QASPER 更适合 4 项、PubMedQA-1k 只有"可比性"1 项胜出。**

### 六之四、能否两个数据集都保留？（建议：可以，但必须主副分明）

**技术成本很低**——已探测确认 PubMedQA-1k 可免费获取（`ori_pqal.json`，2.58 MB，1,000 条，无需注册），且其 `CONTEXTS` 本身就是**按段落组织的句子列表**，能直接映射到我们的 `paragraphs.parquet` schema：

| 字段 | PubMedQA | 映射到本项目 |
|---|---|---|
| `CONTEXTS` | list[段落]，每段是句子列表 | → `paragraphs`（`para_id = {pmid}#p{i}`） |
| `QUESTION` | 问题 | → `qa.question` |
| `final_decision` | yes / no / maybe | → `qa.answer_type`（三分类，不是四分类） |
| `LONG_ANSWER` | 长答案 | → `qa.free_form_answers` |
| `MESHES` / `YEAR` | 元数据 | → metadata |

M1 侧增量约 1 个脚本、1–2 小时（cleaner / tokenizer / flattener / manifest / validate 全部复用）。

**但有一个决定性限制必须先说清：PubMedQA 原始任务是"给定摘要回答问题"，不是"从语料里检索"。**

`CONTEXTS` 是随题目一起给出的上下文。要把它变成检索任务，**必须自己造语料**——CG-RAG 正是这么做的（自建 7,849 篇引文图）。不造语料，M2 的 FAISS 索引和 M3 的图检索**全都无事可做**；造语料，就是一个独立的数据工程项目。

而且它的段数极少（每篇约 2–3 个 `CONTEXTS`）：

- **M1 的 Chunk Size/Overlap 实验无从做起**（每篇 1–3 个 chunk）
- **M3 的图检索在 3 段摘要上没有意义**（文内边几乎为 0，跨文档边需自造语料）
- **M5 算不出段落级 evidence recall**（只有 yes/no/maybe 标签，没有证据标注）

### 建议：**主一副一**，而不是双主线

| | QASPER | PubMedQA-1k |
|---|---|---|
| 定位 | **主战场** | **冒烟测试 + 泛化性附录** |
| M1 投入 | P4–P7 全部 | 一条 data-prep 脚本，不做 chunk 网格 |
| M2 投入 | 全库索引 | 另建一个小索引（约 1/20 规模） |
| M3 投入 | 图检索主实验 | **零**（明确不做图） |
| M5 投入 | 段落级 recall 全套指标 | 一条 answer-level 指标（需 M5 同意） |
| 在报告里的作用 | 主结果 | "同一 pipeline 在第二领域也能跑通" |

**判断线**：如果距截止还剩不到两周、而 QASPER 的主结果还没跑完，**砍掉 PubMedQA**。双数据最常见的失败模式是"两个都跑了，两个都不深"。


**一个必须写进报告的诚实提醒**：PubMedQA-1k 上，1,000 条摘要彼此之间未必有稠密引用关系（原始数据集根本没有引用信息）。如果它作文档间的图是稀疏的，那 CG-RAG 在 PubMedQA 上的增益**很可能主要来自"图编码器充当了学习型重排器"**，而不是"引文结构带来了多跳相关性"。这一点我们无法验证（作者没公布图）。反过来说，**QASPER 因为有 55.5% 的多段落证据，"结构是否有用"这个因果问题反而问得更干净**。


## 七、已知风险 / 坑

1. **v0.3 的 tgz 只含 train+dev**，test（416 篇）标注不在包里，要做 test 评测需另外申请。
2. **沙箱的批量删除守卫**（每轮 50 次上限）会让 `Remove-Item -Recurse` 和 pip 大包安装中途崩溃；装包用 `--ignore-installed --no-deps` 小批量，或 `pip download` + 只补缺失文件。
3. `evidence` 有 9.11% 是 `FLOAT SELECTED:`（图表证据），正文里不存在，**必须从 recall 分母剔除**。
4. 残余 265 条不可对齐 evidence 是 **S2ORC 子章节标题**（`A ::: B` 形式），P6 要用"取该小节全部段落"处理，别当失败。
5. 单段最长 3,636 token，`chunk_size < 3,636` 时硬切必然发生，`is_truncated` 会被触发。
6. **指标定义错一次，结论就反一次。** A8 第一版把「overlap 复制段落」当成「段落被撕裂」，直接得出"overlap 有害"。**改动指标定义后必须重跑并复核结论方向**，不能只改代码不改文档。
7. **`--report-only` 只重画报告、不重跑网格**：改指标口径时用它，7 分钟能省下来。但注意报告里的叙述性数字是脚本算的，硬编码的例外组合（如某个 size/overlap 不存在）会直接越界报错——已改成防御式取值。
8. **pysbd 每段新建 `Segmenter` 是全域最热的调用**（61k 段）。改成模块级单例后输出逐字节一致（已用 17,653 个 chunk 的 `text_sha1` 全量比对验证），网格整体从预计 25 分钟降到 7.5 分钟。
9. **贪心的 tie-break 必须固定顺序。** 候选集是 `set`，Python 字符串哈希按进程随机化，同分时先遇到谁就选谁 → 同一份代码两次运行 `oracle_recall@5` 会差 0.2pp。改成 `sorted(cands - selected)` 后两次运行完全一致。**报告里的数字必须可复现，这不是洁癖。**
10. **`--limit-papers` 的冒烟运行会覆盖正式 CSV/JSON。** 已被一次 40 篇的检查静默覆盖过（24 行 → 6 行）。现在带 `limit` 的运行会写独立文件名。
11. **PowerShell 工具单条命令上限 10 分钟**（`timeout` 设再大也会被夹到 600000ms）。全量网格 + 消融要拆成多次运行，或用 `--no-report` 分片跑再合并（见 `scripts/dev/merge_grid.py`）。
12. **`json.dumps(default=str)` 会把 ndarray 静默变成 repr 字符串。** `"['a' 'b']"` 是合法 JSON、类型却是 str，肉眼极难发现。所有 jsonl 一律走 `jsonable()` + `allow_nan=False`。
13. **`NaN` 不是合法 JSON。** parquet 的 NULL 读回 pandas 可能是 `pd.NA`/`NaN`，直接 `!= None` 比较会让每篇论文的首尾 chunk 都被报成"邻居链断裂"（虚报 2,335 条）。比较前必须归一化。
14. **数据契约不写成可执行代码就等于没写。** A9 的校验器上线后立刻抓出 3 个真实缺陷（重复 span、jsonl 类型错、NaN 写出），这些在人工 review 里都没被发现。
