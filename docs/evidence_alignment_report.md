# A4 证据对齐报告

> 生成时间：2026-10-31T18:19:35+00:00
> 配置：`use_fuzzy=True`，阈值 88，section 规则 True

## 1. 逐 split 结果

| split | questions | answerable | unanswerable_only | textual_evidence | mapped | align_rate_pct | questions_with_gold | questions_zero_gold | float_only_questions |
|---|---|---|---|---|---|---|---|---|---|
| dev | 1005 | 945 | 60 | 2555 | 2553 | 99.92 | 890 | 115 | 35 |
| train | 2593 | 2322 | 271 | 3823 | 3823 | 100 | 2101 | 492 | 139 |

## 2. 证据条目的对齐路径分布

| evidence_level | count | pct |
|---|---|---|
| paragraph | 5883 | 83.84 |
| float | 639 | 9.11 |
| section | 274 | 3.9 |
| paragraph_norm | 219 | 3.12 |
| fail | 2 | 0.03 |

- **文本证据对齐率 99.97%**（分母 = 全部 evidence 条目 − float − 空）
- 仍有 **2** 条无法对齐

## 3. 关键校验

| check | value |
|---|---|
| 可回答但 gold 为空的题目数 | 276 |
| 只有图表证据（float_only）的题目数 | 174 |
| 抽取式答案片段总数 | 4476 |
| 其中能定位到段落 | 4343 (97.0%) |
| 语料索引缺失的论文数 | 0 |

> ⚠️ 有 **276** 条可回答问题的 gold 为空，M5 计算 recall 时应把它们单独列出，不能直接当 0 分（否则会系统性拉低指标）。字段 `zero_gold_reason` 给出了原因，分布如下：

| zero_gold_reason | answerable | unanswerable_only |
|---|---|---|
| align_fail | 2 | 0 |
| float_only | 174 | 0 |
| no_evidence | 100 | 331 |

- 因此**可用于计算 recall 的题目数 = 2991** （3,598 − 607 条 gold 为空）

| reason | 含义 | 建议处理 |
|---|---|---|
| `no_evidence` | 标注者没选任何证据段（绝大多数是 yes/no 题，如「他们用了预训练模型吗？」）。**注意含大量 unanswerable 题** | **从 recall 分母剔除**，只在答案级指标里出现 |
| `float_only` | 证据是图/表（`FLOAT SELECTED`），正文里没有对应文本 | **从 recall 分母剔除** |
| `align_fail` | 文本证据但确实对不上（仅 2 题） | 人工抽检后可接受 |

## 4. section 规则的作用

- 靠 `section_name` 命中的证据条目：**274** 条
- QASPER 的 `section_name` 本身是 S2ORC 层级路径（`"Experiments ::: Ablation"`），全语料 **15,155 / 61,317（24.7%）** 的段落带 `:::` 层级；这条规则把它们解析成「该小节的**全部段落**」，同时给 M3 免费提供了层级边。

## 5. 字段说明（给 M5）

| 字段 | 说明 |
|---|---|
| `gold_para_ids` | 该题所有标注者证据的段落并集（**recall 的唯一依据**） |
| `evidence_levels` | 本题证据走过的对齐路径，逗号分隔 |
| `is_answerable` | 任一标注者给出非 unanswerable 答案 |
| `is_float_only` | 证据只有图表，正文里不存在 → 应从 recall 分母剔除 |
| `extractive_spans_located` | 抽取式答案片段 + 所在 para_id + 段内偏移（用于算「答案与证据是否同 chunk」） |
| `answer_types` | 本题各标注者的答案类型集合（7.8% 的题不一致，聚合口径需 M5 定） |
