# B 线 · PubMedQA-1k 副线报告

> 目的：**不是**做第二主实验，而是回答一个问题——**这套管道是不是只为 QASPER 定制的？**
> 结论先行：**管道通用，但 PubMedQA 承载不了 chunk 级实验**。定位 = 冒烟测试 + 文档级检索补充，M3 零投入。
> 生成时间 2026-11-01　|　脚本 `scripts/10_prepare_pubmedqa.py`

---

## 1. 一句话结论

PubMedQA-1k 在**默认 512 切分下 976/1,000 篇论文被切成恰好 1 个 chunk**（平均 1.02）。也就是说这里根本不存在"chunk 选择"这个问题——整个检索任务退化成"从 1,000 篇摘要里挑对那一篇"。

这不是数据质量问题，是这个数据集的任务形态决定的。因此：

| 用途 | 判断 |
|---|---|
| 冒烟测试（跑通 切分→检索→融合→生成） | ✅ **值得做**，1,000 条几秒跑完 |
| 文档级检索的补充评测（1,000 候选） | ✅ **可用**，而且正好补上 QASPER 缺的"找对论文"这一环 |
| chunk size / overlap 网格实验 | ❌ **无意义**（每篇只有 1 个 chunk） |
| 图检索（文内边 / 跨文档边） | ❌ **无意义**（3.4 段/篇，文内边几乎为 0） |
| 段落级 evidence recall | ❌ **算不出来**（只有 yes/no/maybe，没有证据标注） |

---

## 2. 实测规模（全部来自脚本，非估计）

| 项 | 值 |
|---|---|
| 题数（= 论文数） | **1,000** |
| 段落数 | **3,358** |
| 总 token（BGE tokenizer） | **307,024** |
| 段落/篇 | mean **3.36** / max 9（分布：3 段 720 篇、2 段 69 篇、4 段 101 篇…） |
| token/篇 | mean **307** / **max 648** |
| 段落 token | mean 91.4 / median 77 / p95 218 / max 455 |
| 判定分布 | yes **552** / no **338** / maybe **110** |
| 数据源 | `raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`（2.58 MB，无需认证） |

**关键数字是 `token/篇 max = 648`。** 连最长的摘要也只有 648 token，而默认 `chunk_size = 512`——**绝大多数论文装得进一个 chunk，剩下的也只差一点**。

---

## 3. 切分退化的量化证据

用**完全相同**的 `chunker.py`（`section_boundary=soft`，`min_chunk_tokens=64`）跑两个 size：

| 配置 | chunks | chunks/篇 | 分布 | 单段 chunk 占比 |
|---|---:|---:|---|---:|
| **512 / 0.2**（项目默认） | **1,024** | **1.02** | 1 chunk: **976 篇**，2 chunks: 24 篇 | 2.0% |
| 128 / 0.2 | 4,230 | 4.23 | 峰值在 4 chunks（291 篇），最多 10 | **51.4%** |

两条推论：

1. **默认配置下切分是空操作。** 976 篇论文的"chunk"就等于论文本身。M1 在 PubMedQA 上没有任何可调的东西。
2. **即使把 size 压到 128，也无法获得多段落 chunk。** 51.4% 的 chunk 只含 1 个段落——因为段落本身平均 91 token，剩下能挤进去的邻居很少。

对照 QASPER：默认配置 **15.10 chunks/篇**（17,653 / 1,169），mean 5,504 token/篇。**两者相差 18 倍**，这不是参数问题，是数据形态问题。

---

## 4. 与 QASPER 的逐项对照

| 维度 | QASPER v0.3（主线） | PubMedQA-1k（副线） | 差距 |
|---|---|---|---|
| 论文 / 题 | 1,169 / 3,598 | 1,000 / 1,000 | 题目密度 3.1 : 1 |
| 段落 | 61,317 | 3,358 | **18×** |
| 正文 token | 6,433,441 | 307,024 | **21×** |
| token/篇 | 5,504 | **307** | **18×** |
| 章节结构 | 13.9 sections/篇，24.7% 带 `:::` 层级 | 单一 "Context"，**0 结构** | — |
| 默认 512 下 chunks/篇 | **15.10** | **1.02** | **15×** |
| 证据标注 | ✅ 段落级 gold（2,991 题可用） | ❌ 只有 yes/no/maybe | 决定性差异 |
| 多段落证据题占比 | 37.2%（1,112 / 2,991） | 不适用 | — |
| 任务形态 | 给定论文找证据段 | 给定摘要做三分类 | 不同任务 |
| 清洗规则命中 | `refs` 58,354 / `strip` 5,772 / … | **仅 `nfkc` 16** | QASPER 规则在此完全空转 |

**最后一行值得单独说**：QASPER 的清洗规则里，处理 S2ORC 占位符（`BIBREF0`）那条命中 58,354 次，在 PubMedQA 上**一次都没触发**——因为 PubMedQA 是已经清洗好的 PubMed 摘要。这说明**清洗配置是数据集相关的**，不能想当然地"通用"。管道代码通用，配置不通用。

---

## 5. 管道通用性验证（B 线真正的价值）

把 PubMedQA 塞进**同一套** `chunker.py` + `03_build_chunks.py` + `05_validate_contract.py`，不改一行下游代码，结果：

```
契约校验：23 PASS / 1 WARN / 0 FAIL
```

- 唯一的 WARN 是我额外加的一列 `decision`（原始 yes/no/maybe），需要文档记录——**契约按设计报了 WARN 而不是默默放过**。
- `spans.paired` / `coverage.content` / `graph.neighbours` / `integrity.text_sha1` 等结构性检查全部通过，说明 `para_id` / `chunk_id` / `para_spans` 的设计**不依赖 QASPER 的字段形态**。

这是 B 线最有价值的产出：**它证明了 M1 交付的 schema 是可迁移的**，而不只是"在 QASPER 上能跑"。

---

## 6. 建议的用法（明确边界）

### 6.1 值得做的：文档级检索冒烟测试

PubMedQA 的退化反而提供了一个干净的任务：**1,000 篇摘要里，能否用问题找到正确的那一篇？**

- 候选池 1,000，随机基线 0.1%，任务不平凡
- 每题**恰好一个**正确文档（不像 QASPER 每题只覆盖本论文，没有跨论文噪声干扰）
- 这是**doc-level Recall@k**，恰好补上 QASPER 缺的那一环（QASPER 原设定是篇内检索）

推荐报告两个数：`doc Recall@1/@5`（检索器）+ `yes/no/maybe 准确率`（端到端生成）。

### 6.2 明确不做

| 不做 | 原因 |
|---|---|
| chunk size / overlap 网格 | 每篇 1 个 chunk，参数无区分度 |
| 任何图检索实验 | 3.4 段/篇，文内边≈0；跨文档边需要自建语料（CG-RAG 自己造了 7,849 篇） |
| 段落级 evidence recall | 没有段落级 gold，分母不存在 |
| 与 QASPER 的表格并排比较 | 任务、指标、语料池都不同，并排会诱导读者直接比 |

### 6.3 如果一定要用它和 CG-RAG 对话

只有当**语料口径**写清楚时才有意义：CG-RAG 把 PubMedQA 扩成 7,849 篇算 Hit@1，我们如果只用 1,000 条摘要，池子小 7.8 倍，**同样方法的 Hit@1 必然更高**。这不是方法更强，是任务更简单。详见 `docs/跨数据集可比性说明.md`。

---

## 7. 复现

```powershell
$PY=C:\Users\zxt123\.workbuddy\binaries\python\envs\ragdata\Scripts\python.exe
$env:PYTHONPATH = "src"

# B1+B2：下载 + 转成统一 schema
$PY scripts/10_prepare_pubmedqa.py --raw-dir data/raw/pubmedqa `
    --processed-dir data/processed/pubmedqa --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json

# B3：用同一套 chunker 建表（两个 size 做对照）
$PY scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 --section-boundary soft `
    --name pubmedqa_512_ov102 --processed-dir data/processed/pubmedqa --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json
$PY scripts/03_build_chunks.py --chunk-size 128 --overlap 0.2 --section-boundary soft `
    --name pubmedqa_128_ov26 --processed-dir data/processed/pubmedqa --out-dir docs `
    --tokenizer data/raw/bge_tok/tokenizer.json

# B4：同一套契约校验
$PY scripts/05_validate_contract.py --processed-dir data/processed/pubmedqa `
    --chunks-dir data/processed/pubmedqa/chunks/pubmedqa_512_ov102 `
    --out docs/contract_validation_pubmedqa.md
```

---

## 8. 产出清单

| 文件 | 内容 |
|---|---|
| `data/raw/pubmedqa/ori_pqal.json` | 原始数据（2.58 MB，gitignored） |
| `data/processed/pubmedqa/paragraphs.parquet` | 3,358 段，与 QASPER **同 schema** |
| `data/processed/pubmedqa/qa.parquet` / `qa_dataset.jsonl` | 1,000 题，`gold_para_ids` 全空（无证据标注） |
| `data/processed/pubmedqa/manifest.json` | 规模统计 + 清洗命中 + 判定分布 |
| `data/processed/pubmedqa/chunks/pubmedqa_{512_ov102,128_ov26}/` | 两套 chunk 表 |
| `docs/contract_validation_pubmedqa.md` | 契约校验 23 PASS / 1 WARN / 0 FAIL |
