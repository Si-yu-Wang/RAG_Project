"""Precomputed contract demo, NOT graph search. Run: python -m examples.custom_retriever"""
import json
from dense_rag import Baseline, PreviewGenerator


class PrecomputedRetriever:
    config = {"method": "precomputed_demo", "corpus_sha256": None,
              "parameters": {"source": "synthetic interface example"}}

    def __init__(self, results_by_question):
        self.results_by_question = results_by_question

    def retrieve(self, question, top_k=5, paper_id=None):
        candidates = self.results_by_question.get(question, [])
        candidates = [c for c in candidates if paper_id is None or c["paper_id"] == paper_id]
        return [{**c, "rank": rank} for rank, c in enumerate(candidates[:top_k], 1)]


def main():
    question = "Which encoder is used?"
    retriever = PrecomputedRetriever({question: [{
        "chunk_id": "demo-a:method:0", "paper_id": "demo-a",
        "text": "The system uses a BGE encoder.", "score": 1.0,
        "metadata": {"entities": ["system", "BGE"], "relation": "uses"},
    }]})
    result = Baseline(retriever, PreviewGenerator()).run({
        "question_id": "custom-demo-1", "paper_id": "demo-a", "question": question,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
