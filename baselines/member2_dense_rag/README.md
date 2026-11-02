# 通用 RAG Baseline：可替换检索器与生成器

版本 0.3.0。`Baseline` 统一执行 **检索 → 上下文组织 → 生成 → 结果记录**，不依赖具体检索算法。内置 DenseRetriever 使用 BGE + FAISS；图检索、融合检索通过相同协议接入。

本包包含通用流程和真实 Dense 实现，**不包含真实图检索或融合算法**。自定义示例使用预计算结果演示适配方式。preview 生成器不调用 LLM，不能用其答案做质量评测。

## 上游 data-v1.0 数据接入（新增）

已对接 [RAG_Project](https://github.com/awuliwu/RAG_Project) 的 [数据契约](https://github.com/awuliwu/RAG_Project/blob/5432b484cd2ed9a8dfd5bbf4909151fd8a0a8f82/docs/contract/data_contract.md)，验证快照为 `5432b484cd2ed9a8dfd5bbf4909151fd8a0a8f82`。原始仓库未修改；本包直接消费其交付文件。

| 接口 | 行为与注意事项 |
|---|---|
| `data.load_chunks(path, split=None, manifest_path=None)` | 支持 Parquet / JSONL，返回 chunks 与来源配置 |
| `DenseRetriever.build(..., split=None, manifest_path=None)` | 直接读上游 chunks；不需要中间格式转换 |
| `evaluation.export_requests(qa_path, output, split=None)` | 仅在评测侧从 QA 导出 question_id/paper_id/question/split，不导出任何答案或 gold |
| `data.load_requests(path, split=None)` | 推理侧读取无答案请求；拒绝 qa* 文件及带答案字段的输入 |
| `evaluation.paragraph_recall(qa_rows, predictions, top_k=5)` | 评测侧段落 macro recall；无可评题返回 null，不是 0 |

**字段以契约为准：chunk 正文用 `text`（已清洗）；段落表才使用 `text_clean`。** 不能将 paragraphs.parquet 直接当 chunks 输入。所有章节、split、n_tokens、para_ids、para_spans、前后邻居和 config_hash 在检索结果中保留在原来的顶层字段。Parquet 中的 para_spans JSON 字符串自动解码为数组；不修改 ID，不重新切块，不去掉 overlap 引起的文本重叠。

`para_spans` 是段内坐标；`char_start/char_end` 是清洗后论文扁平文本坐标。适配器检查 span 与 para_ids 一一对应、边界顺序、text_sha1 和配置一致性；不读取 paragraphs 验证每个 span 的上界，这仍需上游完整校验器。

同目录 manifest 自动读取，也可通过 `--upstream-manifest` 指定。索引只保留 config_name/config_hash/config/tokenizer 及 manifest 文件 hash，不将其中 gold 统计作为检索特征。不同切块配置不可混入一个索引；chunk_id 仅在同一配置中稳定。两个数据集可能使用相同 config_hash，因此它不是数据集唯一标识，结果还记录语料文件 SHA-256。

### QASPER：按 dev 跑基线

以下命令在本包目录执行，把 `$DATA` 改成上游仓库绝对路径：

```powershell
$DATA = "D:/projects/RAG_Project"

# 评测侧准备无答案请求；这是唯一读取 QA 的步骤
.venv/Scripts/python -m dense_rag.evaluation --qa "$DATA/data/processed/qa.parquet" --split dev --output artifacts/qasper_dev_requests.jsonl

# 索引只消费公开 chunks；过滤 dev 不依赖 QA
.venv/Scripts/python -m dense_rag.cli build --chunks "$DATA/data/processed/chunks/size512_ov102_soft/chunks.parquet" --split dev --index artifacts/qasper_dev_index

# 推理仅消费已去除标签的请求；默认 preview，可按下文替换真实 LLM
.venv/Scripts/python -m dense_rag.cli run --index artifacts/qasper_dev_index --questions artifacts/qasper_dev_requests.jsonl --split dev --top-k 5 --output artifacts/qasper_dev_predictions.jsonl
```

QASPER 只有 train/dev 标注。检索指标先把 `retrieved_chunks[*].para_ids` 合并去重，再与 gold_para_ids 求召回，最后对题取平均。只纳入 zero_gold_reason 为空字符串的题；全集为 2,991，dev 等子集应动态统计。不要将无 gold 题当零分，不要用 gold_chunk_ids 做跨配置 recall。辅助评测函数遇到缺失预测、重复 ID、K 不足等情况报错，避免静默减少分母；它不是完整答案评分器。

```python
from dense_rag.data import read_records
from dense_rag.evaluation import paragraph_recall

qa_dev = [q for q in read_records("上游/data/processed/qa.parquet") if q["split"] == "dev"]
predictions = read_records("artifacts/qasper_dev_predictions.jsonl")
metrics = paragraph_recall(qa_dev, predictions, top_k=5)
```

### PubMedQA：相同数据接口、不同回答任务

```powershell
.venv/Scripts/python -m dense_rag.evaluation --qa "$DATA/data/processed/pubmedqa/qa.parquet" --output artifacts/pubmed_requests.jsonl
.venv/Scripts/python -m dense_rag.cli build --chunks "$DATA/data/processed/pubmedqa/chunks/pubmedqa_512_ov102/chunks.parquet" --index artifacts/pubmed_index
.venv/Scripts/python -m dense_rag.cli run --index artifacts/pubmed_index --questions artifacts/pubmed_requests.jsonl --task pubmedqa --output artifacts/pubmed_predictions.jsonl
```

`task="pubmedqa"` 使用 yes/no/maybe 分类提示词并记录 task 与 prompt_version；当前不强制解析或重写模型输出，正式评测必须检查标签是否合法。preview 仍返回占位文本，不能用于分类准确率。无上下文时返回 maybe 并标记 no_evidence，评测时应单列该状态。PubMedQA 没有段落 gold，因此段落 recall 为 null。其实际导出数据包含 test split，不能套用 QASPER 的 train/dev 规则。

输入按数据集、切块配置、split 分开管理。上游 n_tokens 只是其 tokenizer 的计数，不能假设与任意 embedding 模型完全一致；默认 BGE 的 512 上限仍可能截断其他数据，需结合 tokenizer 和特殊 token 检查。

## 1. 模块与入口

| 模块 / 文件 | 职责 | 接口 |
|---|---|---|
| 数据与索引 / core.py | 校验 chunks、保存及加载 Dense 索引 | DenseRetriever.build/load |
| BGE / core.py | 正文和问题向量化 | BGEEncoder.encode(texts, query=False) |
| 检索协议 / pipeline.py | 统一检索输入、证据输出和配置 | Retriever.config、retrieve(...) |
| 生成协议 / pipeline.py | 统一消息输入和答案输出 | Generator.config、generate(messages) |
| 通用流程 / pipeline.py | 上下文预算、契约校验、计时和记录 | Baseline(...).run(request) |
| CLI / cli.py | 内置 Dense 的建库、检索、批量运行 | build、retrieve、run |

pipeline.py 仅依赖 Python 标准库。`from dense_rag import Baseline` 不加载 NumPy、FAISS 或 BGE。CLI 目前仍是 Dense 专用；图检索和融合检索使用 Python API 接入。

## 2. 安装与快速开始

在本目录运行，推荐 Python 3.12：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python -m unittest discover -s tests -v
```

上述为包含 Dense 依赖的完整安装。若只验证通用接口，可在本目录用普通 Python 运行预计算检索示例，无须安装 FAISS 或下载模型：

```powershell
python -m examples.custom_retriever
```

构建并运行 Dense：

```powershell
.venv/Scripts/python -m dense_rag.cli build --chunks examples/chunks.jsonl --index artifacts/my_index
.venv/Scripts/python -m dense_rag.cli retrieve --index artifacts/my_index --paper-id demo-a --question "Which encoder is used?" --top-k 5
.venv/Scripts/python -m dense_rag.cli run --index artifacts/my_index --questions examples/questions.jsonl --output artifacts/my_preview.jsonl
```

建库目标目录、输出文件必须尚不存在，防止覆盖旧实验。首次 Dense 运行需下载模型。附带的 artifacts/demo_index 可以直接加载，语料为人工示例。

## 3. 数据输入接口

### chunks.jsonl

UTF-8 编码，每行一个对象：

```json
{"chunk_id":"p1:method:0","paper_id":"p1","text":"The method uses BGE.","metadata":{"section":"Method","paragraph_index":0}}
```

| 字段 | 要求 |
|---|---|
| chunk_id | 非空字符串，在同一语料和切块配置中唯一；同配置的不同检索方法使用相同证据 ID |
| paper_id | 非空字符串，与问题中的论文 ID 完全一致 |
| text | 非空字符串，存放可检索正文 |
| metadata | 可选 JSON 对象，保留章节、原文位置、图路径等 |

空语料、重复 ID、关键字段缺失会报错。不会自动切块或清洗。BGE 默认编码上限为 512 tokens，超长正文会截断；应按实际 tokenizer 控制 chunk 长度。标准答案不应写入检索正文。

### Baseline 请求 / questions.jsonl

```json
{"question_id":"q1","paper_id":"p1","question":"Which encoder is used?"}
```

question 必须非空；paper 模式必须提供 paper_id。global 模式不会按该字段过滤。question_id 可省略，但正式批量运行应提供唯一 ID，便于关联结果。CLI 拒绝带 gold/answer 等标签的输入；Python Baseline 仅把问题传给检索器，不转发额外标签。正式调用仍应传无答案请求。

## 4. Retriever：替换检索器

统一协议如下，不必继承基类，实现同名方法和 config 即可：

```python
class MyRetriever:
    config = {
        "method": "graph",
        "corpus_sha256": None,  # 正式运行建议填写真实语料 hash
        "parameters": {"max_hops": 2},
    }

    def retrieve(self, question: str, top_k: int = 5,
                 paper_id: str | None = None) -> list[dict]:
        # 调用自己的算法，将节点/路径映射为下述证据格式。
        ...
```

Retriever 是类型提示用的 Protocol。config 必须为可 JSON 序列化、没有 NaN/Infinity 的字典。建议包含 method、corpus_sha256、parameters；代码不强制某种算法参数。不要放模型、图连接、集合或密钥。

标准返回值：

```json
[{"chunk_id":"p1:method:0","paper_id":"p1","text":"The method uses BGE.","score":0.8,"rank":1,"metadata":{"section":"Method"}}]
```

**使用注意：**

* 先按 paper_id 限定候选，再取 Top-K。None 表示全库；不能全库取 K 后再过滤。
* 最多返回 K 条，候选不足就少返回，无结果返回 []。不能重复补齐或退回其他论文。
* chunk_id 必须唯一，rank 按返回顺序从 1 连续编号。Baseline 保留该顺序，不重新按 score 排序。
* score 必须为有限 Python int/float。Dense 使用余弦相似度；图/融合使用自己的分数定义，不能跨算法直接比较。没有原生分数时可用 1/rank，并在 config 中标注。
* 所有字段必须可以写入 JSON；NumPy 数值、节点对象等应在适配器中转为普通值。
* 图检索最终须提供可读 text 和可追溯 chunk_id，不能只返回节点 ID。metadata 可以保留实体、关系或路径。
* 当前生成上下文只使用 ID 与 text，不自动把 metadata 图路径传给 LLM。若需路径直接参与生成，须扩展上下文策略和提示词版本。

Baseline 会拒绝超 K、重复 ID、越过论文范围、非法 rank/score、空正文等结果。算法负责相关性排序，异常直接传播。可运行示例见 [custom_retriever.py](examples/custom_retriever.py)，该示例只演示预计算结果适配，不实现图算法。

## 5. Generator：替换生成模型

输入为 role/content 消息列表：

```python
class MyGenerator:
    config = {"provider": "custom", "model": "实际模型版本", "temperature": 0.0}

    def generate(self, messages: list[dict]) -> dict:
        # 在此调用实际 SDK，并返回下面的结构。
        ...
```

返回格式：

```python
{"answer": "The method uses BGE [p1:method:0].", "usage": {"total_tokens": 123}}
```

answer 必须为字符串。usage 可省略，提供时须为 JSON 对象。config 记录非敏感模型配置。失败应抛出异常，不能把错误信息冒充答案。

内置 HTTP 适配器：

```python
from dense_rag import HTTPGenerator

generator = HTTPGenerator(
    endpoint="http://localhost:8000/v1/chat/completions",
    model="实际模型名称", temperature=0.0, max_tokens=512, timeout=120,
)
```

endpoint 是完整的 Chat Completions 兼容地址。密钥从 RAG_API_KEY 环境变量获取，未设置时不发送 Authorization。默认超时 120 秒，没有自动重试；服务必须支持发送的 model/messages/temperature/max_tokens 参数。

CLI 真实生成：

```powershell
$env:RAG_API_KEY = "你的密钥"
.venv/Scripts/python -m dense_rag.cli run --index artifacts/my_index --questions examples/questions.jsonl --output artifacts/my_llm_run.jsonl --generator http --endpoint "http://localhost:8000/v1/chat/completions" --llm-model "实际模型名称"
```

非兼容服务通过 generate 适配。更换生成模型不需要重建 Dense 索引；更换 embedding 模型或正文数据需要重建。

## 6. Baseline：统一编排与注意事项

```python
from dense_rag import Baseline, DenseRetriever, PreviewGenerator

retriever = DenseRetriever.load("artifacts/demo_index")
baseline = Baseline(
    retriever=retriever,  # 可以换成实现上述协议的 graph/fusion 检索器
    generator=PreviewGenerator(),
    top_k=5, context_chars=12000, scope="paper",
)
result = baseline.run({
    "question_id": "q1", "paper_id": "demo-a",
    "question": "Which encoder is used?",
})
```

top_k 和 context_chars 必须是正整数，scope 只支持 paper/global；task 默认 qa，也可设 pubmedqa。按排序逐个加入证据，最后一段可能截断。预算单位是 Unicode 字符，不是 token，也不包含问题、system prompt、JSON 包装；应按生成模型窗口调整。

无证据时跳过生成器并返回 no_evidence。有证据时 status 为 ok，只表示流程完成，不代表答案正确。提示词要求引用 chunk_id，但未自动验证引用真实性或判断问题是否可回答。

每次调用复制配置和检索结果，后续修改后端对象不会改变已返回记录。计时不含运行前配置快照、模型初始化、索引加载与写盘；retrieval 包含查询编码/检索及返回值校验和复制；context 包含消息组织；generation 包含服务调用与输出校验。

## 7. 结果格式与版本迁移

当前 prediction schema_version 为 **2.0**。索引 manifest 仍为 **1.0**，已有 Dense 索引可以继续使用。

| 字段 | 含义 |
|---|---|
| question_id/question/paper_id | 原请求信息 |
| status/answer | 状态与答案 |
| retrieved_chunks | 完整检索结果，含 metadata、score 和 rank |
| contexts | 实际提供给模型的正文及 truncated 标记 |
| context_chunk_ids | 实际上下文 ID，不代表已验证的答案引用 |
| usage | 生成服务用量 |
| timing_ms | retrieval/context/generation/total |
| config.retriever | 检索方法、语料版本、专属参数 |
| config.generator | 生成模型配置 |
| config.top_k/scope/context_chars/prompt_version | 公共参数 |

旧版字段迁移（内置 Dense）：

| 原路径 | 新路径 |
|---|---|
| config.encoder | config.retriever.parameters.encoder |
| config.corpus_sha256 | config.retriever.corpus_sha256 |

Graph/Fusion 不要求有 encoder，消费者应先读取 config.retriever.method。旧 Python 导入 dense_rag.core.Baseline 保留；通用调用推荐从 dense_rag 或 dense_rag.pipeline 导入，避免加载 Dense 依赖。

CLI 逐行保存，异常时停止，保留已完成记录。没有自动断点续跑；重跑使用新文件名，并自行根据 question_id 对齐。

## 8. Dense 实现注意事项

默认 BGE 为 BAAI/bge-small-en-v1.5，revision 固定为 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a，长度 512 tokens，batch size 32。文档直接编码，查询添加固定指令。向量 L2 归一化后使用 IndexFlatIP 精确余弦检索。

index.faiss、chunks.jsonl、manifest.json 必须一起交付。载入校验 hash、模型配置、维度与数量。更换模型、revision、语料或切块策略要重新建索引，仅加载可信索引。更换模型还应检查查询指令与最大长度，不可仅修改模型名。

当前建库将全部 chunks 和向量放入内存，论文子索引按需缓存；大规模数据需要另外评估内存和吞吐。构建失败可能留下不完整目录，使用新目录重试。

## 9. 验证和边界

验证包含 Dense 回归、没有 manifest/encoder 的自定义检索器、非法返回值和全库范围等测试，见 [验证记录](docs/验证记录.md)。原理与设计补充见 [详细说明](docs/Member2_详细说明.md)。

人工示例及两套上游真实数据的小样本均已实际运行 BGE 检索。预计算适配示例、HTTP 模拟测试不代表真实图算法或 LLM 服务已联调。正式质量评测需真实数据与生成服务，并排除 provider=preview 的答案。
