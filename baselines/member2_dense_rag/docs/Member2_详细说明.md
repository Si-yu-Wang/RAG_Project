# 通用 Baseline 与 Dense Retrieval 设计说明

版本 0.3.0；更新日期 2026-11-02。完整接口签名、字段约束、运行命令及示例以 [README](../README.md) 为准。本说明聚焦模块原理和设计边界。

## 1. 总体设计

Baseline 是算法无关的 RAG 流程控制器，接收 Retriever 和 Generator 两个对象。检索器返回统一的证据片段；流程校验其契约，按预算构造上下文，调用生成器并输出可追溯结果。它不读取 Dense 索引、不调用 embedding，也不依赖 encoder 或 manifest 属性。

pipeline.py 只使用标准库。Dense 实现放在 core.py，公共包通过延迟加载暴露 DenseRetriever 和 BGEEncoder。这样图检索调用方可仅使用通用流程，不必在导入时安装 Dense 运行库。完整 pip 安装目前仍默认安装 Dense 依赖，尚未划分 optional extras。

## 2. 检索器边界

Retriever 协议包括 config 和 retrieve(question, top_k, paper_id)。config 是可序列化的算法记录；方法参数放在 parameters 内。Baseline 每次运行复制 config，避免后续参数变化改写历史记录。

证据在同一语料和切块配置内以 chunk_id 为统一标识，以 paper_id 限定范围，以 text 提供可直接生成的内容。rank 决定最终上下文顺序，score 保留方法自身的相关度定义。算法不同，分数不能假定同量纲；融合应自行约定归一化或使用排名方法。

统一接口不强制检索算法，但强制基础不变量：结果不超 K、ID 不重复、rank 连续、分数有限、论文范围正确、文本非空、整体可 JSON 序列化。错误在生成前抛出，避免把错误证据送入 LLM。

图检索应先完成实体匹配、路径搜索、证据定位，再转成标准 chunks。融合检索应先完成合并、去重和排序。原始节点对象或多个重复 chunk 不应直接交给 Baseline。examples/custom_retriever.py 只是预计算结果示例，并不执行图算法。

## 3. Dense Retrieval 原理

BGE 将正文与问题编码为固定长度向量。正文编码在建库阶段执行，查询编码在检索阶段执行。默认英文模型使用固定 revision，查询单独添加检索指令，正文不加指令。

向量归一化后，内积等于余弦相似度。FAISS IndexFlatIP 穷举候选向量，返回精确 Top-K。论文模式先建立该论文候选子索引，再检索，避免全库 Top-K 后过滤造成漏检。子索引按需缓存，因此首次请求和后续请求可能有不同耗时。

索引以三件套保存：index.faiss 储存向量，chunks.jsonl 保存向量行号对应证据，manifest.json 记录模型、数据 hash 和文件 hash。三者必须一致，不能单独替换正文。校验用于检测损坏与误配，不认证来源安全。

向量存储约占 4×N×d 字节，不含正文和子索引缓存。当前全量内存建库适合基础实验；大语料可另行实现流式编码或近似索引，但必须明确方法变化。

## 4. 上下文和生成

Baseline 按检索顺序加入文本，累计不超过 context_chars。保留完整 retrieved_chunks，同时记录真正送入生成器的 contexts，便于区分检索缺失和上下文截断。预算按字符而非 token 计算，不含提示词和 JSON 包装，因此需要根据实际模型窗口调整。

system prompt 要求根据证据回答、引用 chunk ID，并在证据不足时说明无法回答。证据被描述为来源文本，而非操作指令。当前不自动验证模型引用、事实正确性或回答类型，也没有基于分数训练拒答判别器。

Generator 只需提供 config 和 generate(messages)，返回 answer 与可选 usage。HTTPGenerator 适配兼容 Chat Completions 的服务；其他接口可自行包装。PreviewGenerator 明确返回占位文本。无上下文时不调用任何生成器。

metadata 原样保留在 retrieved_chunks，但不进入当前提示词。如果希望路径或关系本身直接成为证据，应明确修改上下文构建及 prompt_version，而不是假设图 metadata 自动生效。

## 5. 实验输出与兼容

prediction schema 从 1.0 升至 2.0，因为配置从 Dense 专属 encoder 变为通用 config.retriever。Dense 模型配置现在位于 config.retriever.parameters.encoder。索引 manifest 仍是 1.0，旧索引不用重建。

每题记录问题 ID、答案、全部检索证据、实际上下文、配置和四类耗时。retrieval 包含后端检索及返回值检查，context 为上下文组织，generation 为生成调用及结果检查，total 为这些阶段总耗时。不包含初始化、下载、加载索引和写盘，不能据此声称端到端冷启动延迟。

比较不同检索方法时，应统一语料、问题、检索范围、生成器、K 和上下文预算。按原始证据映射计算 Recall 等检索指标；按所选数据集协议处理多参考答案、不可回答问题和答案类型。正式评分应拒绝 preview 输出，失败不能静默从统计中删除。

CLI 目前提供 Dense 的 build/retrieve/run。自定义检索器通过 Python API 注入。异常采取 fail-fast，批量运行已写入的行保留，不提供自动续跑或自动重试。

## 6. 技术依据与当前边界

BGE 编码及查询指令参考 [BAAI 模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5) 和 [BGE 官方文档](https://bge-model.com/bge/bge_v1_v1.5.html)。归一化向量内积的余弦检索依据见 [FAISS MetricType](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)，索引说明见 [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。

已提供真实 BGE + FAISS 人工样例索引、通用接口示例和测试。尚未实现实际图检索/融合算法，未运行真实 QASPER 全量评分或真实 LLM 生成。实际验证范围详见验证记录。


## 7. data-v1.0 对接变化

新增 data.py 负责 Parquet/JSONL 读取、段落 span 解析、公开字段检查和上游配置校验。chunk 使用已经清洗的 text，paragraph 才使用 text_clean；保留顶层 para_ids、section_path、邻居、坐标与 config_hash，不将这些结构信息丢进不可访问的临时对象。

新增 evaluation.py 隔离 QA 标签文件。它可以在评测侧导出仅含题目和 ID 的请求，并按段落、逐题 macro 计算检索召回。构建索引完全不依赖 QA；推理入口拒绝直接使用 qa* 或带标签的请求。列名检查不能替代防止逻辑泄漏的实验纪律。

prediction schema 仍为 2.0，新增可选 split、config.task 与 Dense config.data 来源信息；旧索引缺少 data 时返回空字典。上游 manifest 的 gold 汇总不写入索引配置。已验证旧 Dense 接口与通用检索协议保持兼容。

PubMedQA 使用单独分类提示词，无 gold 段落时不计算 recall。QASPER 全集可用题数为 2,991；任何子集按 zero_gold_reason 动态统计。当前仅提供检索指标辅助函数，没有完整的拒答/分类/自由文本答案评分器。

参照仓库快照与实际运行范围见 upstream_validation.json 和验证记录。真实数据只做了两篇论文的小样本 BGE 建库与检索，没有全量性能或质量结论，也没有真实 LLM 回答。
