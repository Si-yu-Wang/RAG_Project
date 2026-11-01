# Chunking 统计 · `pubmedqa_128_ov26`

> 2026-11-01T05:05:48+00:00　|　tokenizer `tokenizers-file:tokenizer.json`
> 配置：size=128，overlap=26，min=64，section=soft，abstract=含

## 1. 规模

| split | chunks | tokens | avg_tokens |
|---|---|---|---|
| test | 4230 | 439483 | 103.9 |

- chunk 总数 **4,230**，覆盖 1000 篇论文
- token 总量 439,483（正文 307,024）→ **索引放大 1.431×**
- 长度：mean 103.9 / median 109 / p95 127
- 硬切（`is_truncated`）chunk：57

## 2. 不变量校验

| check | value |
|---|---|
| chunks | 4230 |
| duplicate_chunk_ids | 0 |
| empty_chunks | 0 |
| over_budget_chunks | 0 |
| chunks_spanning_sections | 0 |
| paragraphs_checked | 3358 |
| paragraphs_with_boundary_gaps | 10 |
| paragraphs_with_content_gaps | 0 |
| broken_neighbour_links | 0 |

## 3. gold 映射（A7）

| metric | value |
|---|---|
| questions | 1000 |
| questions_with_gold_paras | 0 |
| questions_gold_unreachable | 0 |
| co_split_denom | 0 |
| co_split_hit | 0 |
| co_split_rate | 0 |

> `co_split_rate` = 抽取式答案与它的证据段落落在**同一个 chunk** 的比例。这个指标比平均 recall 更能解释坏 case：证据在、答案被切走 → 模型必错。
