"""Validated multi-cutoff retrieval scoring; gold is confined to evaluation."""
from collections import Counter
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


from .data import read_records as rows, sha256 as digest


def scores(gold, ranked_paragraph_sets, k):
    retrieved = set().union(*ranked_paragraph_sets[:k]) if ranked_paragraph_sets else set()
    recall = len(gold & retrieved) / len(gold)
    reciprocal_rank = next((1 / rank for rank, paras in enumerate(ranked_paragraph_sets[:k], 1)
                            if gold & paras), 0.0)
    return recall, reciprocal_rank


def evaluate(qa_path, predictions_path, index_path, split='dev', ks=(1, 3, 5, 10)):
    require(bool(ks) and all(type(k) is int and k > 0 for k in ks), 'K values must be positive integers')
    ks = sorted(set(ks))
    qa = [q for q in rows(qa_path) if q['split'] == split]
    predictions = rows(predictions_path)
    require(bool(qa), 'No QA in selected split')
    qmap = {q['question_id']: q for q in qa}
    pmap = {p['question_id']: p for p in predictions}
    require(len(qmap) == len(qa), 'Duplicate QA ID')
    require(len(pmap) == len(predictions), 'Duplicate prediction ID')
    require(qmap.keys() == pmap.keys(), 'Missing or extra predictions; evaluate the complete selected split')
    folder = Path(index_path)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    require(digest(folder/'chunks.jsonl') == manifest['files']['chunks.jsonl'], 'Index chunks checksum mismatch')
    chunks = rows(folder/'chunks.jsonl')
    cmap = {c['chunk_id']: c for c in chunks}
    require(len(cmap) == len(chunks) == manifest['count'], 'Index count mismatch')
    paper_counts = Counter(c['paper_id'] for c in chunks)
    paper_paras = {}
    for c in chunks:
        paper_paras.setdefault(c['paper_id'], set()).update(c['para_ids'])
    configs = {json.dumps(p['config'], sort_keys=True) for p in predictions}
    require(len(configs) == 1, 'Mixed run configurations')
    config = predictions[0]['config']
    require(config['scope'] == 'paper', 'This evaluation expects within-paper retrieval')
    require(config['top_k'] >= max(ks), 'Insufficient retrieval depth')
    require(config['retriever']['corpus_sha256'] == manifest['source_sha256'], 'Corpus provenance mismatch')
    require(config['retriever'].get('data', {}) == manifest.get('data', {}), 'Index split/config provenance mismatch')
    per_question = []
    excluded = Counter()
    for qid, q in qmap.items():
        p = pmap[qid]
        require(p['question'] == q['question'] and p['paper_id'] == q['paper_id'], 'Question alignment mismatch')
        require(p.get('split') == split, 'Prediction split mismatch')
        require(p['status'] == 'ok', 'Non-success prediction requires investigation')
        hits = p['retrieved_chunks']
        require(len(hits) == min(config['top_k'], paper_counts[q['paper_id']]), 'Incomplete Top-K')
        require(len({h['chunk_id'] for h in hits}) == len(hits), 'Duplicate retrieved chunk')
        ranked = []
        for rank, h in enumerate(hits, 1):
            require(h['chunk_id'] in cmap, 'Unknown retrieved chunk ID')
            c = cmap[h['chunk_id']]
            require(h['rank'] == rank and c['paper_id'] == q['paper_id'] == h['paper_id'], 'Rank/paper mismatch')
            require(h['para_ids'] == c['para_ids'] and h['text'] == c['text'], 'Evidence differs from indexed corpus')
            ranked.append(set(c['para_ids']))
        if q['zero_gold_reason'] != '':
            excluded[q['zero_gold_reason']] += 1
            continue
        gold = set(q['gold_para_ids'])
        require(bool(gold), 'Eligible question has empty gold')
        all_paper_paras = paper_paras.get(q['paper_id'], set())
        require(gold <= all_paper_paras, 'Gold paragraph not present in paper corpus')
        per_question.append({str(k): scores(gold, ranked, k) for k in ks})
    metrics = [{'k': k,
                'macro_paragraph_recall': statistics.mean(r[str(k)][0] for r in per_question) if per_question else None,
                'mrr': statistics.mean(r[str(k)][1] for r in per_question) if per_question else None} for k in ks]
    latencies = sorted(p['timing_ms']['retrieval'] for p in predictions)
    return {
        'metric_schema_version': '1.0', 'split': split, 'total_questions': len(qa), 'eligible_questions': len(per_question),
        'excluded_by_reason': dict(excluded), 'metrics': metrics,
        'definitions': {
            'recall': 'Per-question union of para_ids from top-K chunks intersected with gold_para_ids, divided by gold size; macro average.',
            'mrr': 'Reciprocal rank of first chunk overlapping at least one gold paragraph within top-K; zero if none; macro average on the same eligible questions.',
            'eligibility': 'zero_gold_reason is empty; no-gold questions excluded, not scored zero.',
        },
        'retrieval_ms': {'mean': statistics.mean(latencies), 'median': statistics.median(latencies)},
        'config': config, 'index_count': manifest['count'],
        'provenance': {'qa_sha256': digest(qa_path), 'predictions_sha256': digest(predictions_path),
                       'index_manifest_sha256': digest(folder/'manifest.json')},
        'limitations': ['Within-paper retrieval, not cross-paper retrieval.',
                       'Paragraph overlap counts as retrieval relevance even when a chunk contains only part of a paragraph.',
                       'No answer quality or LLM evaluation; preview output is permitted only for retrieval metrics.',
                       'All cutoffs are prefixes of the same recorded Top-K run, not separate generation experiments.'],
    }



def render_report(report):
    lines = ["# 检索评测报告", "", f"Split：{report['split']}；运行问题：{report['total_questions']}；"
             f"有效分母：{report['eligible_questions']}。", "",
             "排除原因：" + json.dumps(report['excluded_by_reason'], ensure_ascii=False), "",
             "| K | 段落级 Macro Recall@K | MRR@K |", "|---|---:|---:|"]
    for item in report['metrics']:
        recall = 'N/A' if item['macro_paragraph_recall'] is None else f"{item['macro_paragraph_recall']:.2%}"
        mrr = 'N/A' if item['mrr'] is None else f"{item['mrr']:.4f}"
        lines.append(f"| {item['k']} | {recall} | {mrr} |")
    lines += ["", "Recall：Top-K chunks 覆盖的 para_ids 去重后与 gold_para_ids 求交集，除以该题 gold 数量，再对题取平均。",
              "", "MRR：首个覆盖任意 gold 段落的 chunk 排名倒数；前 K 无相关结果为 0。与 Recall 使用相同有效分母。",
              "", "只纳入 zero_gold_reason 为空的题；无可评题时为 N/A，不能当作 0。",
              "", "此报告评估论文内检索，使用完整 retrieved_chunks，并非截断后的 contexts。不代表 LLM 答案质量。",
              "", "各 K 为同一检索运行的前缀；只覆盖段落部分文本也按段落命中计。",
              "", "机器可读 JSON 包含完整运行配置、输入哈希与检索耗时。"]
    return '\n'.join(lines) + '\n'
