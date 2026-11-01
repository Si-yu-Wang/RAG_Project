# Chunking 统计 · `size512_ov102_hard`

> 2026-11-01T04:46:09+00:00　|　tokenizer `tokenizers-file:tokenizer.json`
> 配置：size=512，overlap=102，min=64，section=hard，abstract=含

## 1. 规模

| split | chunks | tokens | avg_tokens |
|---|---|---|---|
| dev | 5414 | 1609635 | 297.3 |
| train | 18437 | 5634342 | 305.6 |

- chunk 总数 **23,851**，覆盖 1169 篇论文
- token 总量 7,243,977（正文 6,433,441）→ **索引放大 1.126×**
- 长度：mean 303.7 / median 295 / p95 507
- 硬切（`is_truncated`）chunk：29

## 2. 不变量校验

| check | value |
|---|---|
| chunks | 23851 |
| duplicate_chunk_ids | 0 |
| empty_chunks | 0 |
| over_budget_chunks | 0 |
| chunks_spanning_sections | 0 |
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
| co_split_hit | 1813 |
| co_split_rate | 93.36 |

> `co_split_rate` = 抽取式答案与它的证据段落落在**同一个 chunk** 的比例。这个指标比平均 recall 更能解释坏 case：证据在、答案被切走 → 模型必错。
