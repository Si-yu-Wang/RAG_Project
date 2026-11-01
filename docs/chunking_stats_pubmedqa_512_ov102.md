# Chunking 统计 · `pubmedqa_512_ov102`

> 2026-11-01T05:58:03+00:00　|　tokenizer `tokenizers-file:tokenizer.json`
> 配置：size=512，overlap=102，min=64，section=soft，abstract=含

## 1. 规模

| split | chunks | tokens | avg_tokens |
|---|---|---|---|
| test | 1024 | 310190 | 302.9 |

- chunk 总数 **1,024**，覆盖 1000 篇论文
- token 总量 310,190（正文 307,024）→ **索引放大 1.01×**
- 长度：mean 302.9 / median 300 / p95 455
- 硬切（`is_truncated`）chunk：0

## 2. 不变量校验

| check | value |
|---|---|
| chunks | 1024 |
| duplicate_chunk_ids | 0 |
| empty_chunks | 0 |
| over_budget_chunks | 0 |
| chunks_spanning_sections | 0 |
| paragraphs_checked | 3358 |
| paragraphs_with_boundary_gaps | 0 |
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
