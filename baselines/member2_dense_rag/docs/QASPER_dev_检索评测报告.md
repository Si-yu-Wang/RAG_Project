# QASPER dev：Dense Retrieval 检索评测

评测日期：2026-11-02。结果来自本机完整 dev Top-10 运行，不是示例数据。

## 数据与检查

- dev 请求与预测均为 1,005 条，question_id 完整对应且无重复。
- 计入指标的题数：890（zero_gold_reason 为空字符串）。
- 排除 115 题：no_evidence 78、float_only 35、align_fail 2；没有将这些题作为零分。
- 每条预测的题目、论文 ID、split、返回数量、排名及 chunk 正文/para_ids 均与请求和索引核对。
- 索引：3,940 chunks；仅在问题所属论文内检索。
- 模型：BAAI/bge-small-en-v1.5，revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a。
- FAISS IndexFlatIP，归一化向量余弦相似度；chunk 配置 size512_ov102_soft，config_hash 9febc4ee8cb3。

## 结果

| K | 段落级 Macro Recall@K | MRR@K |
|---|---:|---:|
| 1 | 33.70% | 0.4213 |
| 3 | 63.00% | 0.5569 |
| 5 | 77.33% | 0.5836 |
| 10 | 94.37% | 0.6006 |

检索计时均值约 10.15 ms/题，中位数约 10.12 ms/题。这是当前机器该次运行记录，包含查询处理、检索及结果校验，不含模型初始化、索引加载或写盘；不是跨机器性能保证。

## 指标定义

Recall@K：将前 K 个 chunks 的 para_ids 取并集，与该题 gold_para_ids 相交，除以 gold 段落数量，最后对 890 题作 macro 平均。

MRR@K：chunk 只要覆盖至少一个 gold 段落，就视为相关；取前 K 名中第一条相关 chunk 的排名倒数，无相关项计 0，最后对同样的 890 题作平均。该口径衡量首条相关 chunk 的位置，不要求它覆盖全部证据。正式对比实验应固定这一相关性判定规则。

所有 K 都截取同一 Top-10 检索结果的前缀。使用完整 retrieved_chunks 评估，不使用被字符预算截断的 contexts，因此这些结果不等于生成器实际可见证据的召回率。

## 可以得出的结论与限制

该次 dev 评测中，增加 K 能提高标准证据段落的平均覆盖率；Top-5 为 77.33%，Top-10 为 94.37%。不能据此声称答案准确率达到相同比例，也不能在没有同口径对照实验时声称优于图检索等方法。

本实验是已知论文内检索，不是跨论文开放检索。段落部分内容出现在 chunk 中也按段落命中计数，遵循上游段落级规则；不表示模型看到完整段落或一定能回答。部分论文候选较少，较大 K 可能覆盖大部分甚至全部正文。

生成器使用 preview；尚未进行真实 LLM 回答、拒答或答案质量评测。dev 是验证集，不应将本结果称为独立测试集成绩。继续调参时应记录选择过程，避免把 dev 调参结果包装为未参与调参的数据表现。

## 复现与交付

- `dense_rag/retrieval_metrics.py`：包内评测实现，经 evaluation score 入口调用；评测侧读取 QA 标签。
- `qasper_dev_metrics.json`：精确数值、运行配置与输入 SHA-256。
- 预测来源：本机模块目录 artifacts/qasper_dev_top10_results.jsonl。
- 自动测试覆盖手工指标样例、macro 聚合、无 gold、重复段落、完整性校验及 CLI 报告生成。

在模块目录中，使用本机仓库根目录 Python 环境执行：

```powershell
..\..\.venv\Scripts\python.exe -m dense_rag.evaluation score --qa ../../data/processed/qa.parquet --predictions artifacts/qasper_dev_top10_results.jsonl --index artifacts/qasper_dev_index --split dev --output artifacts/retrieval_metrics.json
```

输出采用独占创建，已有结果不会被覆盖。
