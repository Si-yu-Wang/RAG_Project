# Chunking 统计 · `size512_ov102_soft`

> 2026-11-01T04:42:51+00:00　|　tokenizer `tokenizers-file:tokenizer.json`
> 配置：size=512，overlap=102，min=64，section=soft，abstract=含

## 1. 规模

| split | chunks | tokens | avg_tokens |
|---|---|---|---|
| dev | 3940 | 1888371 | 479.3 |
| train | 13713 | 6568524 | 479 |

- chunk 总数 **17,653**，覆盖 1169 篇论文
- token 总量 8,456,895（正文 6,433,441）→ **索引放大 1.315×**
- 长度：mean 479.1 / median 495 / p95 511
- 硬切（`is_truncated`）chunk：29

## 2. 不变量校验

| check | value |
|---|---|
| chunks | 17653 |
| duplicate_chunk_ids | 0 |
| empty_chunks | 0 |
| over_budget_chunks | 0 |
| chunks_spanning_sections | 12776 |
| paragraphs_checked | 61317 |
| paragraphs_with_boundary_gaps | 4 |
| paragraphs_with_content_gaps | 0 |
| broken_neighbour_links | 0 |

## 3. gold 映射（A7）

| metric | value |
|---|---|
| questions | 3598 |
| questions_with_gold_paras | 2991 |
| questions_gold_unreachable | 0 |
| co_split_denom | 1942 |
| co_split_hit | 1836 |
| co_split_rate | 94.54 |

> `co_split_rate` = 抽取式答案与它的证据段落落在**同一个 chunk** 的比例。这个指标比平均 recall 更能解释坏 case：证据在、答案被切走 → 模型必错。
