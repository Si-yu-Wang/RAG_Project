# 数据契约 v1.0 · QASPER Chunking 交付

> 维护：Member 1（Dataset & Chunking）　|　生效版本：`data-v1.0`　|　最后更新：2026-11-01
> 本文件是**下游唯一权威口径**。字段含义、坐标系、评测分母、防泄漏红线以本文为准；
> 与 `../PROGRESS.md`（进度）、`../experiments/chunk_grid_results.md`（实验）冲突时，本文优先。

---

## 0. 先看这张表：谁该读什么

| 产物 | M2 稠密检索 | M3 图检索 | M4 融合重排 | M5 评测 | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| `chunks.parquet` / `chunks.jsonl` | ✅ | ✅ | ✅ | ✅ | **公开数据**，无任何答案信息 |
| `paragraphs.parquet` | 可选 | ✅ | 可选 | ✅ | **坐标系原点**，段落级 |
| `qa.parquet` / `qa_dataset.jsonl` | ❌ | ❌ | ❌ | ✅ | **评测金标准**，含答案 |
| `qa_with_gold.jsonl` | ❌ | ❌ | ❌ | ✅ | 加了 `gold_chunk_ids`，本配置专用 |

**红线**：`qa*` 这四份文件只能进 M5 的评测脚本。**不得**出现在 M2 的索引构建、M3 的建图、M4 的融合/重排路径里。理由见 §6。

---

## 1. 文件清单

| 路径 | 行数 | 格式 | 主键 |
|---|---|---|---|
| `data/processed/paragraphs.parquet` | 61,317 | parquet | `para_id` |
| `data/processed/qa.parquet` | 3,598 | parquet | `question_id` |
| `data/processed/qa_dataset.jsonl` | 3,598 | jsonl | `question_id` |
| `data/processed/chunks/<config>/chunks.parquet` | 随配置 | parquet | `chunk_id` |
| `data/processed/chunks/<config>/chunks.jsonl` | 同上 | jsonl | `chunk_id` |
| `data/processed/chunks/<config>/qa_with_gold.jsonl` | 3,598 | jsonl | `question_id` |
| `data/processed/chunks/<config>/manifest.json` | — | json | — |

当前发布配置：**`size512_ov102_soft`**（`size=512`、`overlap=102`、`section_boundary=soft`、`min_chunk_tokens=64`、含摘要），共 **17,653** chunk，`config_hash = 9febc4ee8cb3`。
另有对照配置 `size512_ov102_hard`（23,851 chunk），仅用于消融，不作为默认。

**两种格式是同一份数据的两个视图**，字段与类型完全一致（parquet 里 `para_spans` 存 JSON 文本，因为 pyarrow 无法推断 `(str,int,int)` 元组列；jsonl 里是真正的数组）。**下游读哪一种都行，但不许一边读一边假设另一边不同。**

---

## 2. ID 规范

```
para_id  = "{paper_id}#p{para_idx:04d}"              例：1503.00841#p0019
chunk_id = "{paper_id}#s{sec}#c{idx:03d}"            例：1503.00841#s03#c001
           sec = "abs"（摘要）或两位补零的 section_idx
```

三条必须知道的性质：

1. **`chunk_id` 只在一个配置内稳定。** 换 `chunk_size`/`overlap` 后 `idx` 会重排，`c001` 指向的文本完全不同。**跨配置比较一律用 `para_id`，不要用 `chunk_id`。**
2. **`para_id` 与配置无关**，是全局稳定键。任何"同一份证据在不同切分下"的对比都必须落到它上面。
3. `paper_id` 就是 QASPER 的 arXiv ID（如 `1503.00841`），可以直接拿去查 OpenAlex / Semantic Scholar。

---

## 3. `chunks` 表 · 23 个字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | str | 主键，格式见 §2 |
| `paper_id` | str | 论文（arXiv ID） |
| `split` | str | `train` / `dev`。**没有 test**（官方包不含） |
| `section_idx` | int | 原始 `full_text` 顺序；**摘要 = `-1`** |
| `section_name` | str | S2ORC 原始小节名，可能是层级路径（见 §5.2） |
| `section_path` | str | 把 ` ::: ` 换成 ` > ` 的可读路径，如 `Experiments > Ablation` |
| `sec_key` | str | `abs` 或两位补零的 `section_idx`，用于按节编号 |
| `chunk_idx_in_section` | int | 节内序号，0..n-1 |
| `chunk_idx_in_paper` | int | 篇内序号，0..n-1，**等于阅读顺序** |
| `text` | str | chunk 正文（已清洗，见 `../dataset/cleaning_report.md`） |
| `n_tokens` | int | `text` 的 BGE tokenizer 计数 |
| `char_start` / `char_end` | int | **该篇论文清洗后扁平文本**的半开区间 `[start, end)`。不是 PDF 偏移、不是原始 JSON 偏移 |
| `para_ids` | list[str] | 本 chunk 覆盖的段落，按首次出现顺序，**无重复** |
| `para_spans` | list[[str,int,int]] | 与 `para_ids` **一一对应**：`[para_id, 段内起, 段内止)`，偏移相对**该段落自己的 `text_clean`** |
| `is_abstract` | bool | 是否属于摘要（`section_idx = -1`） |
| `is_truncated` | bool | 是否发生了硬切（单句超预算，按 token 边界切开） |
| `n_sentences` | int | 句子数 |
| `n_sections` | int | 跨了几个 section。soft 边界下可能 > 1 |
| `prev_chunk_id` / `next_chunk_id` | str \| null | **同篇内**前后邻居；篇首 `prev`、篇末 `next` 为 `null` |
| `text_sha1` | str | `text` 的 sha1，用于跨机器比对配置是否一致 |
| `config_hash` | str | 生成配置的哈希，应与 `manifest.json` 一致 |

### 3.1 `para_spans` 是本表最容易被误用的字段

```python
# 从 chunk 取回它覆盖的段落文本
pid, a, b = chunk["para_spans"][0]
para_text = paragraphs[pid]["text_clean"][a:b]     # ← 偏移相对段落，不是相对 chunk
assert para_text in chunk["text"]
```

不变量（由 `scripts/05_validate_contract.py` 强制）：

- `len(para_spans) == len(para_ids)`，且**每个段落只出现一次**
- `0 <= a < b <= len(paragraphs[pid].text_clean)`

> 历史坑：早期版本按**句子**写入 span，一个 4 句的段落会输出 4 条完全相同的记录。当时下游指标用 dict 去重才没出事，但任何人按 `zip(para_ids, para_spans)` 读都会错位。v1.0 已修正。

### 3.2 坐标系的三个层次

```
论文扁平文本 ──char_start/char_end──▶ chunk
   └─ 段落 text_clean ──para_spans[a:b]──▶ chunk 内的片段
```

`char_start/char_end` 是**篇级**坐标，`para_spans` 是**段级**坐标。两者不能混用。

---

## 4. `paragraphs` 表 · 14 个字段

坐标系原点，也是 evidence 对齐的落点。

| 字段 | 说明 |
|---|---|
| `para_id` | 主键 |
| `paper_id` / `split` | 同 chunks |
| `section_idx` / `section_name` / `para_idx` | 位置；**摘要 `section_idx = -1`、`para_idx` 从 0 起** |
| `is_abstract` | 是否摘要 |
| `text_raw` | 原始文本（含 `BIBREF0` 之类占位符） |
| `text_clean` | 清洗后文本，**所有下游一律用它** |
| `char_start` / `char_end` / `n_chars` | 在篇级扁平文本中的位置与长度 |
| `n_tokens` | BGE tokenizer 计数 |
| `offset_map` | cleaned↔raw 的逐字符映射（需要回原文引用时用；多数场景可以忽略） |

- 段落 token：mean **104.9** / median 89 / p95 251 / **max 3,611**
- 空段落在展平阶段**已丢弃**（1,133 个），所以 `paragraphs` 里没有空文本
- section 名含 `:::` 层级的段落：**15,155 / 61,317（24.7%）**

---

## 5. `qa` 表 · 评测金标准

### 5.1 规模与分流（**算指标前必须先读这一节**）

| 类别 | 题数 | 能否进 recall 分母 |
|---|---:|---|
| 有 gold 段落 | **2,991** | ✅ **是** |
| 不可回答，标注者未选证据（`no_evidence`） | 331 | ❌ 否（转入"拒答"指标） |
| 可回答，标注者未选证据（`no_evidence`） | 100 | ❌ 否 |
| 证据仅为图表（`float_only`） | 174 | ❌ 否（正文里不存在） |
| 对齐失败（`align_fail`） | 2 | ❌ 否（可人工抽检） |
| **合计** | **3,598** | **可用分母 = 2,991** |

`zero_gold_reason` 字段直接给出上述原因，`""` 表示有 gold。

> **`is_answerable = False` 恰好 331 题，全部是 `no_evidence`** —— 也就是"不可回答"的题本来就没有证据段，这不是数据缺失，是标注设计。
> 把这 607 题当 0 分会系统性拉低 recall，**而且看起来像是检索出了问题**。这是最容易犯、最难察觉的错误。

### 5.2 字段

| 字段 | 说明 |
|---|---|
| `question_id` | 主键（QASPER 原始 ID） |
| `paper_id` | 该题所属论文（**QASPER 每题绑定一篇论文**） |
| `question` | 问题文本 |
| `answer_types` | 逗号分隔，见 §5.3 |
| `is_answerable` | 是否可回答 |
| `gold_para_ids` | **段落级金标准**，`list[str]`，多个标注者取**并集** |
| `n_gold_paras` | 长度 |
| `zero_gold_reason` | `""` / `no_evidence` / `float_only` / `align_fail` |
| `is_float_only` | 证据只有图表 |
| `free_form_answers` | 自由文本答案 |
| `extractive_spans` | 抽取式答案片段（原文子串） |
| `extractive_spans_located` | 同上 + 已定位的 `para_id` 与段内偏移 |
| `yes_no` | `True` / `False` / `null` |
| `writer_nlp_background` / `writer_topic_background` / `writer_paper_read` / `search_query` | 标注者元数据，**只用于分层分析** |

### 5.3 答案类型分布（题级）

`extractive` 1,968 / `free_form` 777 / `yes_no` 494 / `unanswerable` 359（重叠计首次类型）。

### 5.4 gold 的粒度真相

| 指标 | 值 |
|---|---|
| 每题 gold 段落数 > 1 | **1,112 题（37.2%）** |
| 落在单个 chunk 内的题（本配置） | 700 题（19.5%） |
| 每题 gold chunk 数 | mean **1.76** / median 2 / p95 4 / max 10 |

**这条数据决定了一件事：top-1 检索在 QASPER 上没有意义。** 只取 1 个 chunk，最多只能看全 19.5% 的题的证据。M5 汇报时必须给 top-k。

---

## 6. 防泄漏红线（硬性）

**问题**：如果建图或调参时用到了 gold 标注，指标会虚高，而且**从结果上看不出来**。

| 禁止 | 为什么 |
|---|---|
| 用 `gold_para_ids` / `gold_chunk_ids` 连边 | 图直接编码了答案 → "图检索有效"变成同义反复 |
| 用"是否含 gold 段落"作为 chunk 特征 | 同上 |
| 用 `question` × `gold` 的对应关系训练/调参后再在同一份 dev 上报指标 | 过拟合评测集 |
| 用 `extractive_spans` / `free_form_answers` 做检索 query 扩展 | 等于把答案喂给检索器 |

**已提供的机器化检查**：`FORBIDDEN_IN_PUBLIC`（列名黑名单，见 `src/rag_data/validate.py`）会拒收任何含 `gold` / `answer` / `evidence` / `extractive` / `free_form` / `writer_` 等字样的 public 表列。当前 `chunks` 表已通过。

**但列名检查挡不住逻辑泄漏**。M3 建边时请自查：如果删掉 `qa.parquet` 后整个建图流程跑不起来，那就是泄漏了。

---

## 7. 评测口径（M5 按此实现）

### 7.1 检索指标：以 **paragraph** 为单位

```python
# 唯一正确的 evidence recall 算法
retrieved_chunks = topk(...)                       # list[chunk_id]
hit_paras = set()
for cid in retrieved_chunks:
    hit_paras |= set(chunks[cid]["para_ids"])      # chunk 展开成段落
gold = set(qa[qid]["gold_para_ids"])
recall = len(hit_paras & gold) / len(gold)
```

**不要用 `chunk_id` 直接和 `gold_chunk_ids` 比**——chunk 粒度随配置变，跨配置不可比；而 gold 原本就是段落级的。

- **分母**：`zero_gold_reason == ""` 的 2,991 题
- **聚合**：先算每题 recall，再对题取 macro 平均（不是对所有 gold 段落 micro 平均，否则多证据题会主导）
- **`unanswerable`**：不进 recall 分母，单独评"是否拒答"

### 7.2 `co_split_rate`（生成侧固定损失）

定义：抽取式答案片段与它的 gold 证据段落在**同一个 chunk** 内的比例。
本配置 **94.54%**（1,836 / 1,942）→ 约 5.5% 的题**即使检索完美也必然答错**，因为证据和答案被切到了不同 chunk。

这个数不是检索的问题、不是模型的问题，是切分造成的。**报端到端指标时必须把它单独说明**，否则会被误读成系统缺陷。

### 7.3 跨配置比较的唯一正确姿势

| 想比较 | 用什么 | 不能用什么 |
|---|---|---|
| 两个 chunk 配置的检索质量 | `para_id` 级 recall | `chunk_id` 级 recall |
| 两个 chunk 配置的切分代价 | `paras_torn_pct`、`oracle_recall@k` | 平均 chunk 长度 |
| 检索是否达到理论上限 | `oracle_recall@k`（见 `../experiments/chunk_grid_results.md`） | — |

`oracle_recall@k` 是切分给下游划的天花板：本配置 `@1 = 81.62%`、`@3 = 99.37%`、`@5 = 99.97%`。**任何方法的 recall@k 超过它，一定是算错了。**

---

## 8. 校验：把契约变成可执行的东西

```bash
PY=C:\Users\zxt123\.workbuddy\binaries\python\envs\ragdata\Scripts\python.exe
$env:PYTHONPATH = "src"

# 校验一个配置（结构 + 覆盖 + gold 可达 + 防泄漏）
$PY scripts/05_validate_contract.py --processed-dir data/processed \
    --chunks-dir data/processed/chunks/size512_ov102_soft \
    --out docs/contract/contract_validation.md
```

当前结果：**24 PASS / 0 WARN / 0 FAIL**（见 `contract_validation.md`）。

检查项分三类，任何一项 FAIL 都不要用这份数据：

| 类别 | 检查项 |
|---|---|
| 结构 | 主键唯一、ID 可解析且内嵌 `paper_id`、篇内/节内编号连续、`prev`/`next` 闭环、token 预算、非空 |
| 完整性 | `para_spans` 与 `para_ids` 一一对应、偏移在界内、段落零内容缺口、`text_sha1` 自洽、`config_hash` 与 manifest 一致 |
| 语义/泄漏 | public 表无答案派生列、gold 引用全部可解析、有 gold 的题必能映射到 chunk |

---

## 9. 常见错误清单（按踩坑频率排序）

1. **把 607 道无 gold 的题当 0 分** → 分母必须用 2,991。
2. **用 chunk 级 recall 跨 size 比较** → 一律落 `para_id`。
3. **以为 `para_spans` 的偏移相对 chunk** → 相对段落。
4. **以为 `char_start/char_end` 是 PDF 偏移** → 是清洗后扁平行文本坐标。
5. **把相邻 chunk 的文本重复当 bug 修掉** → overlap 造成的重复是设计意图。
6. **假设 `chunk_id` 跨配置稳定** → 不稳定，`c001` 会变。
7. **假设有 test split** → 官方 v0.3 包只有 train(888) + dev(281)，test(416) 的标注没有。
8. **用 `text_raw` 做检索** → 用 `text_clean`，`text_raw` 含 `BIBREF0` 等占位符。

---

## 10. 版本与变更

- 发布版本：**`data-v1.0`**
- 配置标识：`config_hash = 9febc4ee8cb3`（`chunker_version 1.0.0`）
- 每份 `manifest.json` 记录：清洗配置、chunk 配置、校验结果、gold 统计、tokenizer 后端

**schema 变更流程**：任何字段增删改必须先改本文件 + `src/rag_data/validate.py` 的 `CHUNK_REQUIRED` / `QA_REQUIRED` 与检查项，再跑一次全量校验，然后才更新 tag。**不允许只改代码不改契约。**

---

## 附：最小上手示例

```python
import json, pandas as pd

chunks = pd.read_parquet("data/processed/chunks/size512_ov102_soft/chunks.parquet")

# M2：拿文本去编码
texts = chunks["text"].tolist()
ids = chunks["chunk_id"].tolist()

# M3：文内结构边（同篇内前驱后继 + 同 section 兄弟）
edges = []
for r in chunks.itertuples():
    if r.next_chunk_id:
        edges.append((r.chunk_id, r.next_chunk_id, "adjacency"))
for (paper, sec), g in chunks.groupby(["paper_id", "sec_key"]):
    ids_sorted = g.sort_values("chunk_idx_in_section")["chunk_id"].tolist()
    edges += [(a, b, "section_sibling") for a, b in zip(ids_sorted, ids_sorted[1:])]
# 注：section 层级边可以直接用 section_name 的 ":::" 前缀关系构造，无需额外数据

# M5：evidence recall（唯一正确写法）
para = pd.read_parquet("data/processed/paragraphs.parquet")
qa = pd.read_parquet("data/processed/qa.parquet")
pid2chunks = {}
for r in chunks.itertuples():
    for pid in r.para_ids:
        pid2chunks.setdefault(pid, []).append(r.chunk_id)

eligible = qa[qa["zero_gold_reason"] == ""]          # 2,991 题
# recall = |(展开后的命中段落) ∩ gold_para_ids| / |gold_para_ids|，再对题取平均
```
