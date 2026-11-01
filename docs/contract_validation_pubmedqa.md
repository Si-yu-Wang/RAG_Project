# 契约校验 · `pubmedqa_512_ov102`

> 1,024 chunks / 1,000 题　|　结果：**23 PASS，1 WARN，0 FAIL**

> ✅ 全部结构性与防泄漏检查通过。

## 1. chunks 表

| 状态 | 检查项 | 结果 |
|---|---|---|
| ✅ PASS | `artifact.chunks` | 1,024 chunks from pubmedqa_512_ov102 |
| ✅ PASS | `columns.required` | all required columns present |
| ✅ PASS | `columns.extra` | no extra columns |
| ✅ PASS | `columns.leakage` | no answer-derived column in a public artifact |
| ✅ PASS | `id.unique` | 0 duplicate chunk_id |
| ✅ PASS | `id.format` | all ids match {paper_id}#s{sec}#c{idx} and embed their own paper_id |
| ✅ PASS | `order.in_paper` | chunk_idx_in_paper is 0..n-1 per paper |
| ✅ PASS | `order.in_section` | chunk_idx_in_section is 0..n-1 per section |
| ✅ PASS | `graph.neighbours` | prev/next form a closed chain within each paper |
| ✅ PASS | `size.within_budget` | all chunks <= 512 tokens |
| ✅ PASS | `size.non_empty` | no empty chunk |
| ✅ PASS | `spans.paired` | len(para_spans) == len(para_ids) for every chunk |
| ✅ PASS | `spans.unique` | exactly one span per paragraph per chunk |
| ✅ PASS | `spans.in_bounds` | every span inside its paragraph |
| ✅ PASS | `coverage.paragraphs` | every paragraph appears in at least one chunk |
| ✅ PASS | `coverage.content` | no non-whitespace character lost at chunk boundaries |
| ✅ PASS | `integrity.text_sha1` | sha1 matches text on 500 sampled chunks |
| ✅ PASS | `integrity.config_hash` | config_hash == 9febc4ee8cb3 on every row |

## 2. 评测表（M5 专用）

| 状态 | 检查项 | 结果 |
|---|---|---|
| ✅ PASS | `artifact.qa` | 1,000 questions, 0 with non-empty gold_chunk_ids (= the only valid recall denominator) |
| ✅ PASS | `columns.required` | all required columns present |
| ⚠️ WARN | `columns.extra` | extra (document them): ['decision'] |
| ✅ PASS | `gold.para_refs` | all gold_para_ids exist in the paragraph index |
| ✅ PASS | `gold.chunk_refs` | all gold_chunk_ids exist in this chunk config |
| ✅ PASS | `gold.reachable` | every question with gold paragraphs maps to >=1 chunk |

## 3. 配置

| chunk_size | overlap_ratio | overlap_tokens | min_chunk_tokens | section_boundary | include_abstract | chunker_version | config_hash | tokenizer |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.2 | 102 | 64 | soft | True | 1.0.0 | 9febc4ee8cb3 | tokenizers-file:tokenizer.json |

- `co_split_rate` = 0.0% （0/0）
- 有 gold 段落的题目 0 题，不可达 0 题
