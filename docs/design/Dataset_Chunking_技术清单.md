# Dataset & Chunking —— 技术清单（技术选型 + 实施步骤）

> 负责人：Member 1（周旭涛）
> 分支：`feature/dataset-chunking`
> 数据源：QASPER（Dasigi et al., NAACL 2021；5,049 questions / 1,585 NLP papers）

---

## 0. 我这一层的交付边界

M1 在全组里不是"写个切分脚本"，而是**全组的数据契约提供方**。Dense / Graph / Fusion / Eval 四个模块消费的全部文本单元都来自这里，所以我交付的产物必须满足三条硬要求：

| 要求 | 含义 | 不满足的后果 |
|---|---|---|
| **确定性** | 同一 config + 同一份原始数据 → 逐字节相同的产物 | 下游调参结果不可复现，实验白做 |
| **可回溯** | 每个 chunk 能精确回溯到 `(section, paragraph, 字符偏移)` | Eval 组无法算 evidence recall，整个评测链路断掉 |
| **版本化** | 产物目录带 config hash + git commit | 五个人互相覆盖对方的中间文件 |

**一句话原则：M1 冻结 schema 之后，下游只读不写。**

---

## 1. 三个必须先搞清楚的判断（会决定后面所有设计）

### 1.1 QASPER 的 gold evidence 是**文本串**，不是坐标

这是与"论文式 RAG"（CG-RAG 那种用 section_name + paragraph_numbers 定位）的根本差异。QASPER 每条答案里的 `evidence` 是**从正文逐字复制出来的段落/句子文本**，`highlighted_evidence` 是句子级文本。

后果：**"gold paragraph" 不是现成的，得靠文本对齐反推**。所以流水线里必须单独有一个 **Evidence → Paragraph 对齐**阶段（P6），而且要有 fallback 与人工抽检。这一步做砸了，后面所有 retrieval 指标都会失真——这是本任务真正的技术难点，不是切分。

### 1.2 QASPER **没有引文图**

1,585 篇论文之间没有 reference 关系（`full_text` 里不含参考文献，也没有 paper-paper 边）。所以 Member 3 的 Graph Retrieval **只能建文档内的图**：section→paragraph 层级、相邻 chunk 的邻接、实体共现。

我要主动提供给它三样东西，否则它会照着 CG-RAG 的思路去找跨文档边然后发现一条都没有：
- `section_path`（如 `3 Method > 3.2 Training`）用于建层级边
- `prev_chunk_id` / `next_chunk_id` + `para_ids` 用于建序列边
- `paper_id` 用于**禁止跨论文连边**

### 1.3 Evidence 里混着**图表证据**，天然无法映射到 chunk

部分 `evidence` 条目以 `FLOAT SELECTED:` 开头，指向图/表而非正文段落。这些证据在正文里找不到对应文本，必然映射失败。

处理方式：**单独打标 `evidence_type=float`，从 recall 分母里剔除，但单独报告占比**。如果直接当"映射失败"处理，会把对齐召回率算低，然后你花两天去 debug 一个本来就无解的问题。

---

## 2. 技术选型

| 环节 | 选型 | 理由 | 不选什么 / 风险 |
|---|---|---|---|
| 语言 & 环境 | Python 3.11 + `uv` | 依赖锁快、可复现 | 不用全局 pip |
| 数据加载 | 官方 JSON（`qasper-train-dev-v0.3.tgz`）直读 | HF 的 `datasets` 版本把 `full_text` 压平成嵌套 list，**丢掉 paragraph 级的 `section_name`**，而这是我要用的坐标 | 不用 HF 版本做源；只用它做交叉校验 |
| 数据处理 | `pandas` + `pyarrow` | Parquet 列式，下游按需读列 | 不用 JSONL 存全量 chunk（体积大 3–5×） |
| **Token 计数** | **BGE 的 tokenizer**（`transformers` 加载 `BAAI/bge-large-en-v1.5` 的 XLM-R tokenizer） | chunk 大小必须按**下游 embedding 模型**的 tokenizer 计，否则 BGE 的 512 上限会截断 | ❌ 不用 `tiktoken`（cl100k 与 XLM-R 对学术文本偏差可达 20–30%） |
| 分句 | `pySBD`（主）+ 正则兜底 | 学术文本公式/缩写多，`nltk punkt` 在 `et al.` / `Fig.` / 公式上常出错 | 备选 `spaCy sentencizer`，但对公式不友好 |
| 切分实现 | **自写 `SentenceAwareTokenChunker`** | 需要 paragraph provenance + section 边界 + 句级 overlap 对齐 | ❌ 不用 LangChain `RecursiveCharacterTextSplitter`：按字符切、无 paragraph 映射、overlap 会切在词中间、无法保证不跨 section。可作为**对照组**跑一次，用于说明"通用切分器在这类任务上会掉多少" |
| 模糊匹配（P6） | `rapidfuzz`（`token_set_ratio`） | C++ 实现，10 万条 evidence 秒级 | `difflib` 太慢 |
| 文本归一化 | `unicodedata` NFKC + 自写规则 + offset map | 必须**记录偏移映射**才能回溯原文 | ❌ 不用破坏性的 `re.sub` 直接改串 |
| 实验/画图 | `pandas` + `matplotlib` | 团队统一 | 不用 notebook 做长期产物（结果要入库） |
| Proxy 检索器 | `rank_bm25`（+ 可选 `bge-small-en-v1.5`） | 只用于**相对比较**不同 chunk 配置的可检索性，标注为 proxy | 明确声明：这是 M1 的自测工具，不是 M2 的交付物 |
| 测试 | `pytest` + `hypothesis`（选择性） | 切分器的不变量（覆盖率、不重叠、token 上限）适合 property-based | — |

---

## 3. 数据契约（Schema）

### 3.1 `paragraphs.parquet` —— 全局坐标系

| 字段 | 类型 | 说明 |
|---|---|---|
| `para_id` | str | `{paper_id}#p{全局段序:04d}`，**全项目唯一锚点** |
| `paper_id` | str | QASPER `paper_id` |
| `split` | str | train / dev / test |
| `section_idx` | int | 论文内 section 序号（0 起） |
| `section_name` | str | 原文字段，**可能重名**（同一篇里有多个 "Experiment"） |
| `section_path` | str | 层级路径，给 Member 3 建图用 |
| `para_idx` | int | section 内段序 |
| `is_abstract` | bool | 摘要段（合成段，`section_idx = -1`） |
| `text_raw` | str | 原始文本 |
| `text_clean` | str | 清洗后文本 |
| `char_start` / `char_end` | int | 在**展平后全文**中的偏移 |
| `n_tokens` | int | BGE tokenizer 计数 |

### 3.2 `chunks.parquet` —— 下游唯一的输入

| 字段 | 类型 | 说明 | 谁用 |
|---|---|---|---|
| `chunk_id` | str | `{paper_id}#s{section_idx:02d}#c{chunk_idx:03d}` | 全员（主键） |
| `paper_id` | str | | 全员 |
| `split` | str | | 全员 |
| `section_idx` / `section_name` / `section_path` | | | M3 建图 |
| `chunk_idx_in_paper` | int | | M4 做 DOS / 多样性 |
| `text` | str | 最终喂给下游的文本（已含 overlap 去重后的正文） | M2 / M4 |
| `n_tokens` | int | | M4 算 token 预算 |
| `char_start` / `char_end` | int | 展平全文偏移，**跨 config 的不变量** | M6? / 对齐 |
| `para_ids` | list[str] | **本 chunk 覆盖的段落**，核心字段 | M5 算 recall |
| `para_spans` | list[struct] | `{para_id, char_start, char_end}` 段内精确偏移 | M5 |
| `prev_chunk_id` / `next_chunk_id` | str\|null | 同论文相邻 chunk | M3 |
| `is_abstract` | bool | | M4/M5（摘要类问题要分开看） |
| `is_truncated` | bool | 超长单句被硬切 | M5 排除异常 |
| `text_sha1` | str | 内容指纹，查重复用 | 质检 |

### 3.3 `qa.parquet` —— 评测金标准

| 字段 | 类型 | 说明 |
|---|---|---|
| `question_id` / `paper_id` / `question` | | |
| `answer_type` | enum | `extractive` / `free_form` / `yes_no` / `unanswerable`（多答案时取"任一可答"） |
| `free_form_answers` | list[str] | 全部标注者答案，**不做聚合**，由 M5 决定口径 |
| `extractive_spans` | list[str] | 用于算"答案是否和证据同 chunk" |
| `yes_no` | bool\|null | |
| `gold_para_ids` | list[str] | 对齐结果，**recall 的唯一依据** |
| `n_evidence_total` / `n_evidence_mapped` / `n_evidence_float` | int | 对齐质量审计 |
| `writer_nlp_background` / `writer_topic_background` / `writer_paper_read` | | QASPER 自带，M5 可做分层分析（这是个免费的可解释性素材） |

> ⚠️ **强烈建议：所有跨 config 的指标都以 `para_id` 为准计算，`chunk_id` 只做展示。** 否则 chunk size 一变，指标就不可比——这是这类实验最常见的坑。

### 3.4 `manifest.json`

```json
{
  "config_name": "size512_ov0.2_secthard",
  "chunker_version": "1.0.0",
  "git_commit": "abc1234",
  "source_sha256": "…",
  "bge_tokenizer": "BAAI/bge-large-en-v1.5",
  "params": {"chunk_size": 512, "overlap": 102, "min_chunk_tokens": 64,
             "section_boundary": "hard", "snap": "sentence", "include_abstract": true},
  "stats": {"n_chunks": 0, "n_papers": 0, "avg_tokens": 0, "p95_tokens": 0}
}
```

---

## 4. Chunk ID 规范

```
chunk_id = f"{paper_id}#s{section_idx:02d}#c{chunk_idx:03d}"
例：1909.00319#s03#c007
```

- 用 `#` 而非 `::` 分隔：`paper_id` 本身含 `.`，层级分隔符必须不歧义且便于 `split("#")` 解析。
- **不依赖全局排序**：`section_idx` 与 `chunk_idx` 都是论文内局部序号，所以抽取子集（只跑 dev）时 ID 不变。
- 摘要 chunk：`section_idx = 00`，`is_abstract=True`。
- `text_sha1` 用于检测 overlap 导致的近似重复（注意：overlap 会改变文本，所以只在 `overlap=0` 时能精确去重）。

---

## 5. 实施步骤

### P0 · 工程基建
- 分支（已完成）、目录骨架、`pyproject.toml`（uv）、`ruff` + `black` + pre-commit、`pytest` 骨架、`Makefile`
- `.gitignore` 覆盖 `data/raw|interim|processed`、`*.parquet`、`*.faiss`、日志（已完成）
- **验收**：`ruff check` 零错误；`pytest` 跑通；`git status` 干净

### P1 · 数据获取与 Schema 摸底 ★ 先做这一步，不要急着写切分
- 下载官方 v0.3 到 `data/raw/`（不入库）
- `scripts/00_profile_dataset.py` 输出：
  - 真实字段名/类型/缺失率（**以 dump 结果为准，不要照抄任何文档**）
  - 答案类型分布、每题答案条数分布
  - 论文长度（token）分布、section 数分布
  - `evidence` 中 `FLOAT SELECTED:` 占比、`unanswerable` 占比
  - **同一 split 内 `section_name` 重名率的统计**（决定能不能用名字做主键）
- **验收**：`../dataset/dataset_profile.md` 含 6 张统计表 + 3 张分布图，且我能当面解释每个数字

> 已知的一个坑：官方 README 的统计里 train 是 2,593 题 / 2,675 条答案，dev 是 1,005 题 / 1,764 条答案——**每题的答案条数在两个 split 间不一致**。所以"多答案怎么聚合"必须显式定义，不能靠默认。

### P2 · 清洗（带 offset map）
规则（每条独立可开关，最终以 P1 的统计为依据筛选）：
1. Unicode NFKC 归一（全角→半角、连字符统一）
2. 去零宽字符 / 软连字符
3. **合并 PDF 硬换行**：行尾非标点则与下一行拼接
4. 连字符断词修复：`exam-\nple` → `example`
5. 去页眉页脚、页码行（`^\d+$`、常见 running head）
6. LaTeX 残留：`\cite{...}` / `\ref{...}` 替换为空或占位；`$...$` 内容**保留**（公式符号是重要检索信号）
7. 表格残留行合并（QASPER 里表格被解析成多行文本）

- **红线**：每步删除/替换都写入 `offset_map`（`list[(new_start, old_start, old_len)]`），保证任意清洗后位置可映射回原文。**没有 offset map 就没有 evidence 对齐。**
- **验收**：随机 30 篇人工抽检；`gold evidence 在清洗文本中的命中率` 不低于清洗前

### P3 · 展平为 paragraph 索引
- 产出 `paragraphs.parquet`（见 §3.1）
- **验收（lossless 校验）**：每篇论文按 `char_start` 排序后拼接 `text_clean`，与整篇清洗文本逐字符一致（允许插入的分隔符）；`para` 数 == JSON 段落总数

### P4 · Chunker 实现（核心）

算法：

```
输入: 论文 → paragraphs[]（含 para_id, token 序列, 句边界）
1. 按 section 分组（hard 模式）；section 内把段落按序拼成句流 S = [(sent, para_id, span)]
2. 贪心装箱：从游标 i 起累计 token，直到加上下一句会超过 size
   - 若该句属于新段落且"当前 chunk 已有 ≥ min_chunk_tokens"且剩余空间 < 该句长度 → 在段落边界断开
   - 否则允许跨段落（记录两个 para_id）
3. 输出 chunk，记录 para_ids / para_spans / char_start / char_end
4. 回溯 overlap：从 chunk 末尾向前累计 overlap tokens，**吸附到整句边界**（不切句）
5. 下个 chunk 从吸附点开始；若 overlap ≥ size 则报错（参数非法）
6. 单句本身 > size：按 token 硬切，标 is_truncated=True
```

参数：

| 参数 | 候选 | 说明 |
|---|---|---|
| `chunk_size` | 128/192/256/384/512/768 | BGE 上限 512，>512 必须靠 M2 做 pooling 或截断 → 先问 M2 |
| `overlap` | 0 / 0.1 / 0.2 / 0.3 × size | 用比例，但**产物里存绝对 token 数** |
| `min_chunk_tokens` | 32 / 64 | 防止 "Acknowledgments" 这类产生 5-token 空壳 chunk |
| `section_boundary` | hard / soft | hard 不跨 section，但会产生大量短 chunk；soft 允许跨但惩罚 |
| `snap` | sentence / paragraph | overlap 吸附粒度 |
| `include_abstract` | true | 摘要单独成 chunk（问题只由标题+摘要写成，摘要命中率高） |

- **验收（单测全覆盖）**：① 两次运行产物 sha1 相同；② 全文字符覆盖率 100%（无丢字）；③ 任意 chunk `n_tokens ≤ size`；④ hard 模式下无 chunk 跨 section；⑤ overlap 段落在相邻 chunk 中逐字一致；⑥ 空 chunk 数为 0

### P5 · 契约产物导出 + 校验
- `chunks.parquet` + `manifest.json`
- `scripts/02_build_chunks.py --config configs/*.yaml`
- 校验器 `validate.py`：无空 chunk / 无重复 `chunk_id` / 覆盖率 100% / token 分布无长尾异常 / `para_ids` 全非空且能在 `paragraphs.parquet` 里查到
- **产出 `../contract/data_contract.md`（给下游的接口文档，最高优先级交付物）**

### P6 · Evidence → Paragraph 对齐 ★ 最容易低估
三级匹配，逐级降级：
1. **精确匹配**：归一化后（去多余空白、去标点差异）`evidence == para.text_clean`
2. **包含匹配**：`evidence ⊆ para` 或 `para ⊆ evidence`（句子级 evidence 走这条）
3. **模糊匹配**：`rapidfuzz.fuzz.token_set_ratio ≥ 85` 且唯一最优，否则判失败

分支处理：
- `FLOAT SELECTED:` → `evidence_type=float`，不参与 recall 分母
- `unanswerable=true` → `answer_type=unanswerable`，单独统计
- 匹配失败 → 进 `unmapped_evidence` 列表，**必须在报告里给出失败原因分类与占比**

- **验收**：非 float 证据对齐召回率 **≥ 95%**；未对齐样本随机抽 20 条写出原因；产出 `../dataset/evidence_alignment_report.md`

### P7 · Chunk Size / Overlap 实验（见 §6）

### P8 · 交付与交接
- PR 到 `main`；最优配置在全量 1,585 篇上重跑，打 tag `data-v1.0`
- 给 4 位成员各写 3 行交接说明（读哪个文件 / 用哪些字段 / 注意什么）
- 更新 README 的数据生成命令

---

## 6. Chunk Size / Overlap 实验设计

### 6.1 因变量（全部在 M1 层可独立测得，不依赖 M2/M3）

| 编号 | 指标 | 定义 | 为什么重要 |
|---|---|---|---|
| M1 | `chunks_per_paper` / `avg_tokens` / `p95_tokens` | 规模统计 | 决定索引大小与成本 |
| M2 | `index_blowup` | Σ chunk tokens ÷ 正文 tokens | overlap 的代价，1.0 = 零冗余 |
| M3 | `boundary_split_rate` | gold paragraph 被切进 ≥2 个 chunk 的比例 | 证据完整性 |
| M4 | **`co_split_rate`** | **`extractive_spans` 与其 evidence 落在同一 chunk 的比例** | ★ 最能说明问题：证据在、答案被切走 → 模型必错 |
| M5 | `gold_token_precision` | chunk 内 gold token 占该 chunk token 的比例 | 噪声（distractor）多少 |
| M6 | `oracle_recall@{3,5,10}` | 用 proxy 检索器（BM25 / bge-small），gold chunk 是否进 top-k | 检索天花板 |
| M7 | `sec_per_chunk` | 平均每 chunk 覆盖的 section 数 | hard/soft 边界的直接体现 |

### 6.2 实验矩阵

- **主实验**：`chunk_size ∈ {128,192,256,384,512,768}` × `overlap ∈ {0,0.1,0.2,0.3}` = 24 组，**在 dev（281 篇）上跑**
- **消融**（固定最优 size/overlap）：`section_boundary` hard/soft、`snap` sentence/paragraph、`min_chunk_tokens` 0/64、`include_abstract` on/off、`n_tokens` 用 tiktoken vs BGE tokenizer（**用这张图说服全组统一 tokenizer**）
- 全部在 dev 上定参，**test 只在最后跑一次**（避免过拟合评测集）

### 6.3 交付形式

1. **trade-off 主表**：24 行 × M1–M7，按 `oracle_recall@5` 排序
2. **双轴曲线**：x = chunk_size，左 y = oracle_recall@5，右 y = index_blowup，overlap 分色
3. **Pareto 前沿图**：横轴索引成本、纵轴 co_split_rate，标出非支配点
4. **推荐配置**（给一句话理由，例如"512/0.2 在 recall 上仅比 384/0.2 低 0.8pt，但 chunk 数少 33%，推荐 512/0.2"）
5. `../experiments/chunking_experiments.md` 完整记录 + 原始 CSV 入库

---

## 7. 风险清单

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | Evidence 是文本串，对齐失败率高 | recall 指标全错 | ✅ **P1 实测可对齐率 95.85%**（余下 265 条几乎全是 S2ORC 子章节标题，见 §10） |
| R2 | `section_name` 重名 | 用名字当键会串段 | ✅ **P1 实测重名率 0%**（1,169 篇全部无重复）；仍用 `section_idx` 作主键，成本为零 |
| R3 | `unanswerable` 题占比不低 | recall 分母虚高 → 指标好看但没意义 | 按 `answer_type` 分层报告，不混一个总平均 |
| R4 | tokenizer 不一致（tiktoken vs XLM-R） | 标称 512 实际超长被截断 | 统一用 BGE tokenizer；留 5% 余量 |
| R5 | 表格/公式被解析成碎片 | 切出无意义 chunk、检索噪声 | P2 表格行合并；P1 统计表格残留比例 |
| R6 | hard section 边界产生大量短 chunk | chunk 数暴涨、索引变大 | `min_chunk_tokens` + 短 section 合并；对比 soft 模式 |
| R7 | 论文长度长尾（最长 ~10k tokens，短的只有 2 个 section） | 平均指标掩盖极端 case | 报告分位数而非均值；长尾单独列出 |
| R8 | 多人并行改同一份中间文件 | 结果互相污染 | 产物目录强制带 `config_name`；下游只读不写 |
| R9 | 仓库 `README.md` 目前是 **UTF-16LE** 编码 | git 视为二进制，diff 不可读，多人协作易冲突 | 改存 UTF-8 |

---

## 8. 目录结构（本分支新增）

```
RAG_Project/
├── configs/
│   ├── base.yaml
│   └── grid/size{128..768}_ov{0,0.1,0.2,0.3}.yaml
├── src/rag_data/
│   ├── __init__.py
│   ├── config.py            # pydantic 校验 config
│   ├── loader.py            # 读官方 JSON
│   ├── cleaner.py           # 清洗 + offset_map
│   ├── flattener.py         # → paragraphs.parquet
│   ├── tokenizer.py         # BGE tokenizer 封装 + 缓存
│   ├── chunker.py           # 核心切分器
│   ├── evidence_aligner.py  # evidence → para_id
│   ├── exporter.py          # parquet + manifest
│   └── validate.py          # 契约校验
├── scripts/
│   ├── 00_profile_dataset.py
│   ├── 01_prepare_corpus.py
│   ├── 02_build_chunks.py
│   ├── 03_align_evidence.py
│   └── 04_run_chunk_grid.py
├── tests/
│   ├── test_chunker_invariants.py
│   ├── test_cleaner_offset.py
│   └── test_evidence_aligner.py
├── docs/
│   ├── dataset_profile.md
│   ├── data_contract.md            ★ 给下游
│   ├── evidence_alignment_report.md
│   └── chunking_experiments.md
└── data/                            # gitignored
    ├── raw/ interim/ processed/
```

---

## 9. 需要其他成员确认的 5 件事（开工前发群里）

| 问谁 | 问题 | 为什么卡我 |
|---|---|---|
| M2 | embedding 用 `bge-large-en-v1.5` 吗？它的最大输入是 512 token，chunk 要不要卡在 512 以内 | 决定 `chunk_size` 上限 |
| M3 | 图建在 **chunk** 还是 **paragraph** 粒度？除了 para 邻接与 section 层级，还需要什么边？ | 决定我要不要额外导出边表 |
| M4 | 生成时给 LLM 的候选是 top-k 几、token 预算多少 | 决定实验里 `oracle_recall@k` 取哪几个 k |
| M5 | evidence recall 按 chunk 还是按 para 算？`unanswerable` 进不进分母？多答案怎么聚合 | 决定 `qa.parquet` 的字段语义 |
| 全员 | 是否同意 **dev 调参、test 只在最后跑一次** | 防止全组过拟合 |

---

## 10. P1 实测结论（2026-11-01，train+dev 共 1,169 篇）

完整数据见 [`dataset_profile.md`](../dataset/dataset_profile.md) 与 `profile_stats.json`。**下面每条都改写了本清单前面的某个假设**，请以实测为准。

### 10.1 推翻或修正的假设

| 原假设（§1/§7） | 实测 | 处理 |
|---|---|---|
| `paragraphs` 是 dict，带自己的 `section_name` | ❌ **是纯字符串** `paragraphs: ["...", "..."]` | loader 已改成双兼容；paragraph 的 section 一律继承父级 `section_name` |
| `section_name` 会在同一篇内重名（R2） | ❌ **1,169 篇中 0 篇重名** | 风险解除；仍用 `section_idx` 作主键 |
| evidence 对齐会是主要难点（R1） | ✅ 但比预想简单：**95.85% 可对齐**，其中 92% 是**原文逐字精确匹配** | P6 三级匹配降级为"两级 + 一个特例"，工作量大幅下降 |
| 文本是干净的学术正文 | ❌ 36.83% 的段落含 S2ORC 占位符 | P2 必须加规则，见 10.3 |

### 10.2 关键数字（可直接引用）

| 项 | 值 |
|---|---|
| 篇数 / 问题数 / 答案标注数 | 1,169 / 3,598 / 4,439（train 888 篇·2,593 题·2,675 答案；dev 281 篇·1,005 题·1,764 答案） |
| 正文 token 总量（BGE tokenizer） | **6,270,599**（不含摘要 222,942） |
| 每篇正文 token | mean 5,364 / p50 5,032 / p90 8,272 / p99 17,794 / **max 36,529** |
| 每篇 section 数 | mean 13.9 / **max 49** |
| 总段落数 / 空段落 | 61,281 / **1,133（1.85%）** |
| 段落 token | mean 102 / **max 3,636（单段就可能吃掉整个 chunk 预算）** |
| 每题答案数 | mean 1.23 / max 3，**但 train 1.03 vs dev 1.76，两个 split 差异很大** |
| 每题 evidence 数 | mean 1.95 / p99 12 / **max 74** |

**答案类型（按标注计）**：extractive 52.38% / free_form 23.72% / yes_no 13.90% / **unanswerable 10.00%**。
按"问题"计：**7.8% 的问题在不同标注者之间答案类型不一致**（如 `extractive+free_form` 占 5.36%）——所以"M5 要定聚合口径"不是形式问题，是必须拍板的问题。

### 10.3 两个必须立刻处理的数据问题

**① S2ORC 占位符**：`BIBREF` 49,568 次 / `TABREF` 4,546 次 / `FIGREF` 3,859 次，**22,570 个段落（36.83%）含占位符**。样例：

> `BIBREF0 constructed a network of events using word embedding-derived similarities...`

这些 token 对 BGE 是纯噪声（占 token 预算又不携带语义），对 BM25 是完全无意义的"高频词"。P2 清洗必须把它们替换成占位或删除，并**在实验里量化它带来的差异**。

**② 残余 265 条无法对齐的 evidence，几乎全是「子章节标题」**，形如 `Experimental Studies ::: Comparison with State-of-the-art Methods`、`Data and Models ::: Emotion`。`:::` 是 S2ORC 的层级分隔符——也就是说**有些证据指向"某个小节"，而不是"某一段"**。

> 这条直接改写 P6 的设计：不要把它当失败，应该加**第 4 条规则**——若 evidence 匹配某个子章节标题，则把该子章节的**全部段落**记为该问题的 gold，并打标 `evidence_level=section`。这样对齐率能推到 99% 以上，且语义上是对的。

### 10.4 对后续阶段的具体影响

- **P2**：新增占位符规则；过滤 1,133 个空段落；注意单段最长 3,636 token，`chunk_size < 3,636` 时必然触发硬切，`is_truncated` 标记会被用到。
- **P3**：一篇论文有 17,794 token（p99）→ 展平后的 `char_start/char_end` 是 int32 放得下的，但要按论文重置，不要做全局偏移。
- **P4**：`section` 数最多 49 → hard section 边界会产生大量短 chunk，`min_chunk_tokens` 是必需项而不是可选项。
- **P5**：`unanswerable` 占 10%，Eval 组若把它算进 recall 分母会系统性拉低分数；建议默认**单独报告**。
- **P6**：见 10.3 ②，加 `evidence_level ∈ {paragraph, section, float}` 字段。
- **数据获取**：v0.3 的 tgz **只含 train+dev**，test（416 篇）的标注不在其中。若要做 test 评测需另外申请。

---

## 11. P2/P3 实测结论（已完成）

完整数据见 [`cleaning_report.md`](../dataset/cleaning_report.md)，进度见 [`PROGRESS.md`](../PROGRESS.md)。

### 11.1 与本清单原设计的差异

| 原设计（§5 P2） | 实际 | 说明 |
|---|---|---|
| 清理 `\cite{}` / `\ref{}` 等 LaTeX 宏 | ❌ **QASPER 里根本没有** | S2ORC 已经把引用替换成 `BIBREF12` 这类占位符，规则要打的是**占位符**不是 LaTeX 宏 |
| 占位符直接删掉 | 改为替换成 `[REF]/[TABLE]/[FIGURE]` | 直接删会破坏句子语法（`BIBREF0 constructed a network` → `constructed a network`） |
| `\b(BIBREF)\d+\b` 正则 | 去掉 `\b` | **有占位符被粘在单词后面**（`recommendationBIBREF6`），带 `\b` 会漏掉；替换时按需补一个空格 |
| 保留 `$...$` 内容 | ✅ 保留，但**量化后是个大问题**，见 11.3 | |

### 11.2 产出与验收

- `data/processed/paragraphs.parquet`：**61,317 段 / 6,433,441 tokens**，字段见 PROGRESS.md
- **无损契约全部通过**：重复 `para_id` 0；`flat[char_start:char_end] == text_clean` 切片抽查 200 条 0 错；offset map 覆盖一致性 2,000 条 0 错；NFKC 逐字符等价性 3,000 条 0 违反
- 清洗改动过 **44.78%** 的段落；字符量仅 -0.07%（因为 `BIBREF0`→`[REF]` 长度相近），**但 token 结构变了**——这是本次清洗的真实收益，不在字符数上体现

### 11.3 新增的待决策项：LaTeX 密集段落

**8,896 个段落（14.51%）判为 LaTeX 密集，占用 1,253,511 tokens（全语料的 19.48%）**。判据是段内 `$` ≥ 4 次。样例：

> `$$\nabla _\Theta _{rl}\left(\Theta \right) &=& - \nabla _\Theta \left( \mathbb {E}_{\hat{\tau } \sim p_{\tau }} \left[ R \right] \right) \\ ...`

**接近 1/5 的 token 预算花在行间公式上。** 取舍：

- 剔除：chunk 信噪比显著提升，但含公式的问题（QASPER 里确实有）会失去证据
- 不剔除：`chunk_size=512` 时一个公式段就能吃掉整个 chunk，且 BGE 对 `nabla`/`Theta` 这类串几乎没有语义表示

**已在群里列为 Q5，需要 M2 + M5 一起拍板**；在此之前默认**保留**，不擅自删数据。

---

*本清单只定义 M1 的边界与接口；schema 一旦在群里确认，即冻结为 v1.0，后续变更走 PR 讨论。§10 的实测结论优先于前面的预估值。*
