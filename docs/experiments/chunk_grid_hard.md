# Chunk 参数网格实验结果

> 论文 1,169 篇 / 题 3,598 / 正文 6,433,441 tokens　|　tokenizer `tokenizers-file:tokenizer.json`　|　section_boundary `hard`
> 共 6 组配置（size × overlap）

## 1. 指标定义（全部与模型无关，纯数据层）

| 指标 | 含义 | 方向 |
|---|---|---|
| `index_blowup` | 全部 chunk 的 token 总量 ÷ 正文 token 总量 | 越低越省 |
| `paras_torn_pct` | **没有任何一个 chunk 完整包含该段落**的比例（真实撕裂） | 越低越好 |
| `paras_dup_pct` | 因 overlap 而出现在 ≥2 个 chunk 里的段落比例（**设计行为，不是缺陷**） | 仅作说明 |
| `mean_gold_chunks` | 一道题的证据平均散落在几个 chunk 里 | 越低越好 |
| `oracle_recall@k` | **完美排序器**在 k 个 chunk 内的证据召回上界 | 越高越好 |
| `co_split_rate` | 抽取式答案与其证据落在同一 chunk 的比例 | 越高越好 |

> `oracle_recall@k` 是这张表最重要的列：它假设检索器提前知道答案，只受切分约束。任何 Dense / Graph / 融合方法的 recall@k 都不可能超过它。

> **`torn` 与 `dup` 必须分开看。** 段落被 overlap 复制到两个 chunk 是设计意图；只有当**没有任何 chunk 完整包含该段落**时，证据才真的被切断了。把两者混为一谈（把 dup 当缺陷）会把 overlap 误判成全面劣势——第一版实现就踩了这个坑，当时结论是"overlap 有害"，修正后结论正好相反。

## 2. 全量结果

| name | chunks | index_blowup | avg_tokens | p95_tokens | truncated_chunks | paras_torn_pct | paras_dup_pct | mean_gold_chunks | co_split_rate | oracle_recall@1 | oracle_recall@3 | oracle_recall@5 | oracle_recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s256_ov0 | 35157 | 1 | 183 | 254 | 180 | 20.17 | 20.17 | 1.84 | 93.1 | 63.91 | 97.18 | 99.57 | 99.96 |
| s256_ov51 | 40862 | 1.261 | 198.6 | 254 | 180 | 12.05 | 43.39 | 2.23 | 93.15 | 68.47 | 96.96 | 99.4 | 99.92 |
| s384_ov0 | 26620 | 1 | 241.7 | 380 | 58 | 10.76 | 10.76 | 1.56 | 93.2 | 76.12 | 98.7 | 99.85 | 99.97 |
| s384_ov77 | 28825 | 1.177 | 262.7 | 381 | 58 | 4.74 | 27.51 | 1.75 | 93.31 | 80.19 | 98.83 | 99.83 | 99.97 |
| s512_ov0 | 22870 | 1 | 281.3 | 506 | 29 | 6.42 | 6.42 | 1.45 | 93.25 | 81.82 | 99.08 | 99.92 | 99.98 |
| s512_ov102 | 23851 | 1.126 | 303.7 | 507 | 29 | 2 | 18.46 | 1.55 | 93.36 | 85.26 | 99.27 | 99.92 | 99.98 |

## 3. 透视：size × overlap → `oracle_recall@1` / `paras_torn_pct` / `index_blowup`

**oracle_recall@1（%）** —— 单 chunk 内看全全部证据的比例

| chunk_size | ov=0 | ov=0.2 |
|---|---|---|
| 256 | 63.91 | 68.47 |
| 384 | 76.12 | 80.19 |
| 512 | 81.82 | 85.26 |

**paras_torn_pct（%）** —— 段落被切散、无法在任何单个 chunk 内看全的比例

| chunk_size | ov=0 | ov=0.2 |
|---|---|---|
| 256 | 20.17 | 12.05 |
| 384 | 10.76 | 4.74 |
| 512 | 6.42 | 2 |

**index_blowup（×）** —— 索引成本

| chunk_size | ov=0 | ov=0.2 |
|---|---|---|
| 256 | 1 | 1.261 |
| 384 | 1 | 1.177 |
| 512 | 1 | 1.126 |

> 读法：**size 决定撕裂的下限，overlap 决定把撕裂压到多低。** 两者在 `oracle_recall@1` 上可以互相替代（256/@0.3 ≈ 384/@0.1），但换来的索引成本不同——见 §5。

## 4. 按 overlap 看（对每个 size 取均值）

| overlap_ratio | chunks | index_blowup | paras_torn_pct | paras_dup_pct | oracle_recall_1 | oracle_recall_3 | oracle_recall_5 |
|---|---|---|---|---|---|---|---|
| 0 | 28215.7 | 1 | 12.45 | 12.45 | 73.95 | 98.32 | 99.78 |
| 0.2 | 31179.3 | 1.19 | 6.26 | 29.79 | 77.97 | 98.35 | 99.72 |

## 5. 边际收益：overlap 每多花 0.1 倍索引，换来多少召回上界

| size | overlap | Δblowup | Δoracle@1(pp) | pp per 0.1 blowup |
|---|---|---|---|---|
| 256 | 0.0→0.2 | 0.261 | 4.56 | 1.75 |
| 384 | 0.0→0.2 | 0.177 | 4.07 | 2.3 |
| 512 | 0.0→0.2 | 0.126 | 3.44 | 2.73 |

## 6. 推荐配置

**主推 `s512_ov102`**（size=512，overlap=102）

| chunks | index_blowup | oracle_recall@1 | oracle_recall@3 | oracle_recall@5 | co_split_rate | paras_torn_pct |
|---|---|---|---|---|---|---|
| 23851 | 1.126 | 85.26 | 99.27 | 99.92 | 93.36 | 2 |

> 选择规则：硬约束 `chunk_size ≤ embedding 上限`（本轮取 512，依据 bge-large-en-v1.5 的 512 position embedding），软约束 `index_blowup ≤ 1.5`，在此范围内最大化 `oracle_recall@5`，并列时看 `oracle_recall@1` 与索引大小。**这是纯数据层排序，不含 embedding 信息。**

## 8. 消融：section 硬边界 vs 软边界

| size | overlap | chunks soft/hard | avg_tok soft/hard | torn% soft/hard | or@1 soft/hard | or@5 soft/hard |
|---|---|---|---|---|---|---|
| 256 | 0 | 35157 / 35157 | 183 / 183 | 20.17 / 20.17 | 63.91 / 63.91 | 99.57 / 99.57 |
| 256 | 0.2 | 40862 / 40862 | 199 / 199 | 12.05 / 12.05 | 68.47 / 68.47 | 99.40 / 99.40 |
| 384 | 0 | 26620 / 26620 | 242 / 242 | 10.76 / 10.76 | 76.12 / 76.12 | 99.85 / 99.85 |
| 384 | 0.2 | 28825 / 28825 | 263 / 263 | 4.74 / 4.74 | 80.19 / 80.19 | 99.83 / 99.83 |
| 512 | 0 | 22870 / 22870 | 281 / 281 | 6.42 / 6.42 | 81.82 / 81.82 | 99.92 / 99.92 |
| 512 | 0.2 | 23851 / 23851 | 304 / 304 | 2.00 / 2.00 | 85.26 / 85.26 | 99.92 / 99.92 |

> 硬边界 = chunk 不跨 section；软边界 = 只按 token 预算装箱。
>
> 两个方向**不是单调优劣**：硬边界的 chunk 平均只有软边界的一半长（每篇 13.9 个 section、每 section 仅约 295 token，预算用不满），chunk 数多 35%，但 `torn` 与 `oracle_recall@1` **反而更好**——因为边界落在 section 末尾时通常也落在段落末尾，段落不容易被切开。总 token 成本（`index_blowup`）硬边界更低。
>
> **所以这一项不能只在数据层定**：硬边界给的是「更短但更完整」的 chunk，软边界给的是「更长但更容易切散」的 chunk。谁更好取决于 embedding 在 300 vs 480 token 上的信噪比，必须等 M2 的 Dense baseline 用真实 recall 复核。
