# data/data：GNN-RAG 的 WebQSP 数据

本目录是 GNN-RAG（cmavro/GNN-RAG）在 `gnn/README.md` 里给的 Google Drive `data.zip` 解压后的内容，原样存放，**不要改动里面的文件**（`data_process/` 的转换脚本和 `score_retrieval.py` 都直接读这里）。整个目录被 `.gitignore` 忽略（约 1 GB），clone 后需要自己下载并解压到 `data/data/`。

```
data/data/
├── webqsp/       # 稠密子图（NSM, WSDM 2021），946 MB，整数 id —— GNN-RAG / ReaRev 默认用这个
└── sr-webqsp/    # 稀疏子图（SR, ACL 2022），59 MB，原始字符串
```

两份是**同一批 4,737 道 WebQSP 题**：题目 id、问题文本、答案、topic 实体 MID 完全相同，`*.dep`、`vocab.txt`、`word_emb.npy` 逐字节相同。区别只在每道题配的子图。

---

## 1. 两份数据的区别

| | `webqsp`（稠密） | `sr-webqsp`（稀疏） |
|---|---|---|
| 子图来源 | topic 实体出发 2 跳，PageRank 取前 2,000 个实体 | SR 检索器检索出的关系路径展开 |
| 每题子图大小 | 均值约 4,200 个三元组、1,400 个节点，各题差别不大 | 中位数 31–35 个三元组、25–29 个节点，长尾（均值 75–90 / 61–75） |
| 答案在子图里（有答案的题中） | 95.6%（4,494 / 4,700） | 91.9%（4,321 / 4,700） |
| 存储形式 | 0-based 整数 id，指向 `entities.txt` / `relations.txt` | 原始字符串（MID、关系名、字面值） |
| 额外字段 | — | `paths`（SR 检索出的关系路径） |
| 子图是否依赖问题 | **不依赖**：只由 topic 实体集合决定，topic 相同的题子图完全相同 | 依赖（路径由问题检索得到） |

---

## 2. 规模

| split | 题 | 有答案 | 无答案 | 稠密：答案在子图 | 稀疏：答案在子图 | 稠密空子图 | 稀疏空子图 |
|---|---|---|---|---|---|---|---|
| train | 2,848 | 2,826 | 22 | 2,708（95.8%） | 2,617（92.6%） | 0 | 6 |
| dev | 250 | 246 | 4 | 235（95.5%） | 221（89.8%） | 1 | 2 |
| test | 1,639 | 1,628 | 11 | 1,551（95.3%） | 1,483（91.1%） | 1 | 7 |

每题子图大小（三元组 / 节点，均值 / 中位数）：

| split | 稠密三元组 | 稠密节点 | 稀疏三元组 | 稀疏节点 |
|---|---|---|---|---|
| train | 4,221 / 4,224 | 1,420 / 1,551 | 86 / 33 | 70 / 27 |
| dev | 4,126 / 4,226 | 1,419 / 1,550 | 75 / 31 | 61 / 25 |
| test | 4,306 / 4,415 | 1,448 / 1,638 | 90 / 35 | 75 / 29 |

其它：

- 每题 topic 实体平均 1.02 个（最多 5 个）；dev 有 1 道题没有 topic 实体（它的子图也是空的）。
- 每题答案平均 10.86 个，长尾很重（最多 3,688 个）。答案 `kb_id` 可以是 MID，也可以是字面值（如日期 `1971-10-13`）。
- **dev 是从官方 train 里切出来的**，所以 dev 的 id 是 `WebQTrn-*`；test 的 id 是 `WebQTest-*`。
- `data_process/` 和统一评测用的 test 主分母 **1,551** = test 中「有答案且答案在稠密子图里」的题数。

---

## 3. 文件说明

### 3.1 `{train,dev,test}.json`（两个目录都有）

JSON Lines，每行一道题：

```jsonc
// webqsp（整数 id）
{
  "id": "WebQTrn-9",
  "question": "how old is sacha baron cohen",
  "entities": [99399],                                    // topic 实体，entities.txt 的行号
  "answers": [{"kb_id": "1971-10-13", "text": null}],     // kb_id 是字符串，不是 id
  "subgraph": {
    "tuples": [[1084387, 54, 1128807], ...],              // [头实体 id, 关系 id, 尾实体 id]
    "entities": [...]                                     // 子图节点 id 列表
  }
}

// sr-webqsp（字符串）
{
  "id": "WebQTrn-9",
  "question": "how old is sacha baron cohen",
  "entities": ["m.0c9c0"],
  "answers": [{"kb_id": "1971-10-13", "text": null}],
  "subgraph": {
    "tuples": [["m.0c9c0", "people.person.date_of_birth", "1971-10-13"], ...],
    "entities": ["1971-10-13", "m.0659swt", ...]
  },
  "paths": [["m.0c9c0", ["people.person.date_of_birth", "END OF HOP"]], ...]   // [起点 topic MID, [r1, r2, ..., "END OF HOP"]]，每题约 10 条
}
```

- 稠密数据里所有答案 `kb_id` 都能在 `webqsp/entities.txt` 里找到（已逐题核对，0 缺失），用 `entities.txt` 的行号把答案映射成 id 即可判断命中。
- `answers[].text` 是答案的名字（51,438 个答案里 168 个为 `null`）。**不要用它给子图节点起名**：它只给答案起名，等于泄题。节点名见 §5。
- sr-webqsp 里只有 1 条记录（dev，子图为空）没有 `paths` 字段，读的时候用 `.get("paths", [])`。

### 3.2 其它文件

| 文件 | 目录 | 内容 |
|---|---|---|
| `{split}.dep` | 两者（相同） | JSON Lines：`{"id", "question", "dep": [[词, 词性, 父节点序号(1 起, 0=ROOT), 依存关系], ...]}`，NSM 的问题依存句法 |
| `vocab.txt` | 两者（相同） | 问题词表，6,718 行 |
| `word_emb.npy` | 两者（相同） | `(6718, 300)` float64，与 `vocab.txt` 逐行对应；只在 `--lm lstm` 时用，GNN-RAG 默认的 `--lm sbert` 不需要 |
| `entities.txt` | webqsp | 全局实体表，1,441,420 行，行号即 id |
| `relations.txt` | webqsp | 全局关系表，6,102 行，行号即 id |
| `entity_emb_100d.npy` | webqsp | `(601145, 100)`，**与 `entities.txt` 对不上，别用**（见 §4） |
| `relation_emb_100d.npy` | webqsp | `(567, 200)`，**与 `relations.txt` 对不上，别用** |
| `ents.txt` | sr-webqsp | 子图里出现过的实体，183,433 行 |
| `entities.txt` | sr-webqsp | `= sorted(ents.txt)`，内容相同、顺序不同 |
| `rels.txt` | sr-webqsp | 子图三元组里用到的关系，2,198 行 |
| `relations.txt` | sr-webqsp | SR 的全局 Freebase 关系列表，26,282 行（含 `american_football` 这类 domain/type 前缀） |
| `vocab_new.txt` | sr-webqsp | 与 `vocab.txt` 逐字节相同 |
| `{split}_counter.json` | sr-webqsp | SR 流程的关系频次统计，`[[字符串, 次数], ...]` 降序；训练用不到，而且 dev/test 是反的（见 §4） |

---

## 4. 坑

1. **`webqsp/*_emb_100d.npy` 和词表不对齐。** 实体嵌入 601,145 行 vs `entities.txt` 1,441,420 行，关系嵌入 567 行 vs `relations.txt` 6,102 行。不要传给 `--entity_emb_file` / `--relation_emb_file`。
2. **sr-webqsp 的 `dev_counter.json` 和 `test_counter.json` 是反的**：`dev_counter.json` 是用 `test.json` 算的，`test_counter.json` 是用 `dev.json` 算的。
3. **sr-webqsp 的 `entities.txt` 和 `ents.txt` 是同一组字符串的两种排列**，同一个实体在两个文件里行号不同。要建 id 映射时选定一个文件，别混用。
4. **sr-webqsp 是字符串，GNN-RAG / ReaRev 的 loader 不能直接读**（以下两条是读代码得出的，未实际运行）：
   - ReaRev 的 loader 遇到字符串会崩；GNN-RAG 的 loader 不崩，但会悄悄丢掉 topic 种子。要用的话先把字符串映射成 id。
   - GNN-RAG 的 `gnn/` 代码对路径里带 `sr-` 的目录走另一条分支，需要一个本数据没有提供的 `ent2id.pickle`，会直接失败。
   - 想在 sr-webqsp 上训练，用 SR 仓库自己的 `src/nsm_reader`：把 `*.json` 改名为 `*_simple.json`，并传 `--word_emb_file word_emb.npy`。
5. **sr-webqsp 只有 test 是干净的。** SR 的检索器是在官方 WebQSP train（= 本地 train + dev）上训练的，dev 上的检索质量偏乐观。
6. **两个目录都不给子图节点起名**，而文本 RAG / LLM 方法需要名字。CVT 节点在任何来源里都没有名字。

---

## 5. 节点名

名字不在本目录里，用 `data_process/` 的脚本生成到 `data/raw/webqsp_names/`：

```bash
python data_process/fetch_webqsp_names.py   # 下载 GNN-RAG 的 entities_names.json + HF rmanluo/RoG-webqsp，按 sha256 校验
python data_process/build_mid2name.py       # 合并 → data/raw/webqsp_names/mid2name.tsv
```

- `entities_names.json`（GNN-RAG 发布，约 56 万个 `m.*` MID → 名字）单独用只能覆盖 sr 子图节点出现次数的 41–50%。
- HF `rmanluo/RoG-webqsp` 就是本地稠密 `webqsp` 的同一批子图（三元组相同、顺序相同），只是把名字填上了，覆盖 4,700 道有答案的题。按位置对齐后多出约 8 万个名字。
- 合并后的 `mid2name.tsv` 共 639,943 个名字，两个来源之间 0 冲突。

---

## 6. 怎么用

| 目的 | 用哪份 | 怎么做 |
|---|---|---|
| 跑 GNN-RAG（ReaRev） | `webqsp` | 在 GNN-RAG 的 `gnn/` 目录、单独环境里跑 `main.py ReaRev --data_folder <绝对路径>/data/data/webqsp/ --lm sbert ...`，参数见 GNN-RAG 的 `gnn/README.md` |
| 和 dense RAG 在同一候选池上比检索 | `webqsp` | `data_process/webqsp_to_text.py` 转成仓库的 paragraphs / chunks / qa 格式（输出 `data/processed/webqsp_text/`），见 `data_process/README.md` §3 |
| 统一打分 | `webqsp` | `data_process/score_retrieval.py from-gnn --mode webqsp --gnn-dir data/data/webqsp ...`，见 `data_process/README.md` §5 |
| 跑 SR + NSM | `sr-webqsp` | SR 仓库的 `src/nsm_reader`（见 §4 第 4 条） |

GNN-RAG 环境要点（详见 `data_process/README.md` §2）：

- 需要 `transformers<5`（dense baseline 锁的是 5.x，两者不能共用环境），并修掉 `gnn/parsing.py:81-82` 的 NameError。
- 设 `PYTHONUTF8=1`；`--data_folder` 末尾要带 `/`，路径里不能出现 `sr-` 或 `metaqa`。
- 评测检索时加 `--eps 2` 才会输出完整排序；GNN-RAG 自己打印的 hits/F1 会把无答案题算作命中、且只看 top-1，不要直接引用。

参考数字：

- GNN-RAG 官方发布的 ReaRev 输出，在 test 1,551 道可评题上 Hit@1 = 80.01（按 1,628 道有答案题计为 76.23）。复现命令见 `data_process/README.md` §6。
- sr-webqsp 上不训练、直接沿 SR 预测的 top-1 路径取答案，test F1 = 0.621（SR+NSM 论文为 0.641）。

---

## 7. 本文数字的来源

§1–§3 的题数、答案命中、子图大小、数组形状、文件行数，以及「`.dep` / `vocab.txt` / `word_emb.npy` 两目录相同」「sr `entities.txt == sorted(ents.txt)`」，都是 2026-11-08 直接扫描本目录文件重新算的。§4 第 4 条是读代码得出的，未实际运行；§5、§6 的数字来自 `data_process/` 的此前验证。
