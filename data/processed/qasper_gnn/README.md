# data/processed/qasper_gnn：QASPER 的 GNN-RAG（ReaRev）格式

把 QASPER 转成 GNN-RAG `gnn/` 代码能直接读的数据目录（和 `data/data/webqsp` 同一套文件结构），让 ReaRev 当作**篇内证据检索器**：每道题在本篇论文的 chunk 里排序，候选池与 dense baseline 的 `scope=paper` 完全相同，最后由 `data_process/score_retrieval.py` 和 dense 用同一个 gold、同一组 K、同一个分母打分。

- 生成脚本：`data_process/qasper_to_gnn.py`（设计说明也见 `data_process/README.md` §4）
- 配置：`qasper_inpaper_s1qc_v1`，config_hash `bd94340c8bf7`（`manifest.json`）
- 输入：`data/processed/chunks/size512_ov102_soft/chunks.parquet` + `qa_with_gold.jsonl`（sha256 记在 manifest 里）
- 本目录被 `.gitignore` 忽略（`*.npy` 也全局忽略），clone 后需要按 §7 重新生成

---

## 1. ⚠ 先看：文件名和 QASPER 的划分不是一一对应

| 本目录文件 | 实际内容 | 用途 |
|---|---|---|
| `train.json` | QASPER **train** 中可评题，去掉留出论文 | 训练 |
| `dev.json` | QASPER **train** 里随机留出 15% 论文（133 篇）上的可评题 | 选 checkpoint |
| `test.json` | QASPER **dev** 全部 1,005 题 | 报结果（= dense baseline 报的那个 split） |

- **`dev.json` 不是 QASPER dev**，`test.json` 才是。
- **不要在 `test.json` 上选模型或调参。**
- 留出论文列表在 `heldout_train_papers.txt`（随机种子 20261107）。
- 「可评题」= `zero_gold_reason == ""` 且至少有 1 个 gold chunk。`test.json` 保留了全部 1,005 题，其中 115 题的 `answers` 为空（`no_evidence` 78、`float_only` 35、`align_fail` 2），打分时只算 890 道可评题。

---

## 2. 图的设计

每道题一张独立的小图：

```
            Q::<qid>    ← 虚拟问题根节点 = topic entity / 种子
          /  |  ...  \      关系 0：根 → 每个 chunk 实例
   <qid>|c0  <qid>|c1 ... <qid>|cN    ← 本篇论文的每个 chunk 一个实例节点
        └─1/2─┘ └─1/2─┘                 关系 1/2：阅读顺序相邻
        └──────3──────┘                 关系 3：篇内 BGE 相似（kNN）
```

| 关系 id | 名称 | 含义 |
|---|---|---|
| 0 | `query.chunk.candidate` | 根 → 本篇每个 chunk 实例 |
| 1 | `paper.chunk.next_same_section` | 阅读顺序的下一个 chunk，同一节内 |
| 2 | `paper.chunk.next_new_section` | 阅读顺序的下一个 chunk，跨节 |
| 3 | `paper.chunk.similar` | 论文内 BGE 余弦 kNN（k=3），无向，去掉已经是相邻关系的对 |

- 每条边只存一次，ReaRev 自己会加反向边。
- 图结构只用阅读顺序和 chunk 文本构造，不用任何标签。
- **节点是「每题一份」的 chunk 实例**：同一个 chunk 在同一篇论文的每道题里各有一个节点（key `<qid>|<chunk_id>`），因为节点特征依赖问题（见下）。
- **节点特征** `node_feat.npy`：chunk 实例行 = BGE(q) ⊙ BGE(c)（逐元素乘，按行求和就是 dense 的余弦），根节点行 = 0。ReaRev 本身没有「问题 × 节点文本」这一项，不给特征它只能看到边的类型，结果约等于随机。编码器与 dense baseline 完全一致：`BAAI/bge-small-en-v1.5@5c38ec7c`，L2 归一化，max_length 512。
- **答案** = `gold_chunk_ids`（覆盖 gold 段落的 chunk），写成实例 key。

---

## 3. 规模

| 文件 | 题 | 论文 | 有答案的题 | 每题节点（均值 / 中位数 / p90 / 最大，含根） | 每题边（均值 / 中位数 / 最大） | 每题答案（均值 / 最大） |
|---|---|---|---|---|---|---|
| `train.json` | 1,764 | 698 | 1,764 | 16.25 / 15 / 24 / 97 | 48.9 / 46 / 344 | 2.04 / 9 |
| `dev.json` | 337 | 126 | 337 | 17.45 / 17 / 25.4 / 59 | 53.2 / 49 / 187 | 1.97 / 7 |
| `test.json` | 1,005 | 281 | 890 | 14.79 / 14 / 21 / 63 | 43.7 / 40 / 210 | 2.34 / 10 |

按关系统计的边数（实例级，即各题边数之和）：

| 文件 | 0 根→chunk | 1 同节相邻 | 2 跨节相邻 | 3 相似 |
|---|---|---|---|---|
| `train.json` | 26,905 | 8,883 | 16,258 | 34,242 |
| `dev.json` | 5,545 | 1,798 | 3,410 | 7,185 |
| `test.json` | 13,858 | 4,135 | 8,718 | 17,177 |

全局：49,414 个实体（3,106 个根 + 46,308 个 chunk 实例），4 种关系。论文级的不重复边：相邻 16,484 对，kNN 22,350 对。512 token 的 chunk 大多一节一个，所以跨节相邻（2）比同节相邻（1）多。

---

## 4. 文件说明

| 文件 | 内容 |
|---|---|
| `train.json` / `dev.json` / `test.json` | JSON Lines，GNN-RAG 格式，**纯 ASCII**（见 §6） |
| `entities.txt` | 49,414 行，行号即实体 id。按 train → dev → test 顺序连续编号：train 0–28,668，dev 28,669–34,550，test 34,551–49,413 |
| `relations.txt` | 4 行，行号即关系 id（见 §2） |
| `node_feat.npy` | `(49414, 384)` float32，与 `entities.txt` 逐行对应；根节点行全 0 |
| `id_map.parquet` | 实体 id → `key` / `kind`（`root` 或 `chunk`）/ `file` / `question_id` / `paper_id` / `chunk_id`（根节点为 null）。把 GNN 的节点排序映射回 chunk 用这个表 |
| `heldout_train_papers.txt` | 留给 `dev.json` 的 133 篇 QASPER-train 论文 id |
| `vocab.txt` / `word_emb.npy` | 从 `data/data/webqsp` 原样复制，用来补齐 GNN-RAG loader 期望的目录结构 |
| `manifest.json` | 配置、config_hash、输入 sha256、统计、推荐的 GNN-RAG 参数 |
| `_embed/` | 中间产物：`chunks.jsonl`（17,653 个 chunk 文本）、`queries.jsonl`（3,598 道 QASPER train+dev 问题）及对应的 BGE 向量 `*.npy`（384 维 float32）和 `*.ids.txt`。只有重新 `build` 时才需要 |

一条记录（`train.json`）：

```jsonc
{
  "id": "74b0d3ee0cc9b0a3d9b264aba9901ff97048a897",          // QASPER question_id
  "question": "How did they induce the CFG?",
  "entities": [0],                                           // 只有根节点 Q::<qid>
  "answers": [{"kb_id": "74b0d3ee...|1601.00901#s00#c000",   // kb_id = text = 实例 key
               "text":  "74b0d3ee...|1601.00901#s00#c000"}, ...],
  "subgraph": {
    "entities": [0, 1, 2, ..., 28],                          // 根 + 本篇全部 chunk 实例
    "tuples": [[0, 0, 1], [0, 0, 2], ..., [1, 2, 2], [2, 1, 3], ...]   // [头 id, 关系 id, 尾 id]
  }
}
```

实例 key 的 `|` 后面就是仓库 chunk 表里的 `chunk_id`（`<paper_id>#s<节>#c<序号>`，摘要为 `#sabs#`）。

---

## 5. 怎么用

### 5.1 训练和评测（在 GNN-RAG 的 `gnn/` 目录、单独环境里）

```bash
python main.py ReaRev --name qasper --data_folder <绝对路径>/data/processed/qasper_gnn/ --lm sbert \
  --relation_word_emb False --entity_emb_file node_feat.npy --kge_frozen 1 \
  --entity_dim 50 --num_iter 2 --num_ins 3 --num_gnn 3 --batch_size 8 --num_epoch 50 --eval_every 2 \
  --checkpoint_dir <已存在的目录> --experiment_name qasper_s1qc
python main.py ReaRev <同上参数> --is_eval --load_experiment qasper_s1qc-h1.ckpt --eps 2 --experiment_name qasper_s1qc_eval
```

- 环境：`transformers<5`（不能和 dense baseline 共用），修掉 `gnn/parsing.py:81-82` 的 NameError，设 `PYTHONUTF8=1`。
- 评测必须加 `--eps 2`，才会输出完整排序；默认 0.95 会截断候选，K>1 的指标只能当下限。
- **GNN-RAG 自己打印的 hits/F1 不要用**：它把无答案题算命中、只看 top-1。

### 5.2 打分

```bash
python data_process/score_retrieval.py from-gnn --mode qasper --gnn-dir data/processed/qasper_gnn \
  --info <ckpt_dir>/qasper_s1qc_eval_test.info --out rankings/qasper_dev_gnn.jsonl
python data_process/score_retrieval.py score   --dataset qasper --split dev rankings/qasper_dev_*.jsonl
python data_process/score_retrieval.py compare --dataset qasper --split dev rankings/qasper_dev_dense.jsonl rankings/qasper_dev_gnn.jsonl
```

指标是段落级 Hit@K / Recall@K / MRR@K（K = 1/3/5/10），分母 890，定义与 member 2 的 `dense_rag/retrieval_metrics.py` 一致。

### 5.3 参考数字（test.json = QASPER dev，890 题，此前验证）

| 方法 | Recall@1 | @3 | @5 | @10 | MRR@10 |
|---|---|---|---|---|---|
| 随机（50 次平均） | 14.31 | 38.98 | 58.08 | 86.36 | — |
| 阅读顺序 | 6.38 | 26.20 | 43.74 | 79.79 | — |
| BM25 | 24.90 | 56.20 | 73.79 | 93.18 | — |
| Dense（member 2，BGE 零样本） | 33.70 | 63.00 | 77.33 | 94.37 | 0.6006 |

- 每篇只有十几个 chunk，**@10 基本饱和**（随机就有 86%），主要看 @1/@3/@5 和 MRR。Recall@1 的标准误约 1.5 个百分点。
- 报 GNN 结果时注明：GNN 是在 QASPER train 上**有监督**训练的，dense 是**零样本**。

---

## 6. 坑

1. **JSON 必须是纯 ASCII。** GNN-RAG 的 loader 用平台默认编码打开文件，中文 Windows 上是 cp936，非 ASCII 字符会读错或崩。`qasper_to_gnn.py` 写文件时已强制转义；手动改文件时注意保持。
2. **`--data_folder` 末尾要带 `/`**，且路径里不能出现 `sr-` 或 `metaqa`（GNN-RAG 会走另一条分支）。
3. **不要传 `--relation_emb_file`**（ReaRev 会崩）；**`--name` 不要用 `cwq`**。
4. **`node_feat.npy` 行数必须等于 `entities.txt` 行数**，否则 GNN-RAG 不报错，而是悄悄随机初始化。重新生成后先跑 `check_gnn_folder.py`。
5. **`.info` 的行序与 `test.json` 的记录一一对应**。`from-gnn` 会核对行数；不要手动删改 `test.json` 里的记录（包括 115 道无答案题）。
6. 节点是每题一份的实例，**不能按实体 id 跨题合并**；要回到 chunk 用 `id_map.parquet`，或把 key 按 `|` 切开。

---

## 7. 重新生成

```bash
python data_process/qasper_to_gnn.py export
# 在有 torch + transformers 的环境里（例如 conda 环境 wangsiyu，GPU 更快）
python data_process/embed_bge.py --input data/processed/qasper_gnn/_embed/queries.jsonl --out data/processed/qasper_gnn/_embed/queries --query
python data_process/embed_bge.py --input data/processed/qasper_gnn/_embed/chunks.jsonl  --out data/processed/qasper_gnn/_embed/chunks
python data_process/qasper_to_gnn.py build
python data_process/check_gnn_folder.py data/processed/qasper_gnn
```

- 消融开关（加在 `build` 后）：`--no-features`（不生成 `node_feat.npy`）、`--no-knn`（只有根边和相邻边）。其它参数：`--knn-k`、`--holdout-frac`、`--seed`、`--test-split`。
- 输入 chunk 表或 `qa_with_gold.jsonl` 一变，config_hash 就会变；对比结果时先核对 `manifest.json` 的 config_hash。

---

## 8. 验证状态

- 2026-11-08 重新核对：`check_gnn_folder.py` 通过；实体数、各文件题数、无答案题数、关系分布、`node_feat.npy` 形状、根节点行全 0、各文件实体 id 区间，都与 `manifest.json` 及本文一致。
- 此前验证（`data_process/README.md` §6）：把 `node_feat.npy` 按行求和伪造成 GNN-RAG 的 `.info`，经 `from-gnn` 转换后打分与 dense 完全相同，说明节点 id、实例 key、答案与 chunk 的映射都对。
- **尚未验证**：GNN-RAG 在新环境里真正加载本目录并完成一次训练。
