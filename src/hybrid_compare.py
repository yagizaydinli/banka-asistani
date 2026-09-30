"""Retrieval yöntemlerini aynı 44 soruluk eval setiyle karşılaştırır.

Yöntemler:
  dense            bge-m3 (Hafta 1 baseline)
  bm25             kelime eşleştirme, kelimeler olduğu gibi
  bm25-kok5        kelime eşleştirme, kelimelerin ilk 5 harfi (Türkçe ekleri atmak için)
  hybrid           dense + bm25-kok5, RRF ile birleşik
  hybrid+filtre    hybrid + soruda madde numarası varsa o maddeyi başa al

Kullanım:  python src/hybrid_compare.py     (sonuç: eval/sonuclar/hybrid.json)
"""
import json
from pathlib import Path

from hybrid import BM25Search, rrf, with_filter
from retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "retrieval_sorular.jsonl"
OUT = ROOT / "eval" / "sonuclar" / "hybrid.json"


def metrics(rankings: list[list[str]], questions: list[dict]) -> dict:
    r1 = r5 = rr = 0.0
    ranks = []
    for ranking, q in zip(rankings, questions):
        rank = next((i + 1 for i, cid in enumerate(ranking) if cid in q["gold"]), None)
        ranks.append(rank)
        r1 += rank == 1
        r5 += bool(rank and rank <= 5)
        rr += 1 / rank if rank else 0
    n = len(questions)
    return {"recall@1": round(r1 / n, 3), "recall@5": round(r5 / n, 3), "mrr": round(rr / n, 3), "siralar": ranks}


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    retriever = Retriever()
    ids = [c["id"] for c in retriever.chunks]
    bm25 = BM25Search(retriever.chunks)
    bm25_kok = BM25Search(retriever.chunks, stem=5)

    q_vecs = retriever.model.encode([q["soru"] for q in questions], normalize_embeddings=True)
    dense = [[ids[i] for i in (-(retriever.vecs @ v)).argsort()] for v in q_vecs]
    sparse = [bm25.rank(q["soru"]) for q in questions]
    sparse_kok = [bm25_kok.rank(q["soru"]) for q in questions]
    hybrid = [rrf([d, s]) for d, s in zip(dense, sparse_kok)]
    hybrid_f = [with_filter(q["soru"], h) for q, h in zip(questions, hybrid)]

    methods = {"dense": dense, "bm25": sparse, "bm25-kok5": sparse_kok,
               "hybrid": hybrid, "hybrid+filtre": hybrid_f}
    results = {name: metrics(r, questions) for name, r in methods.items()}

    print(f"{'yöntem':<15} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for name, m in results.items():
        print(f"{name:<15} {m['recall@1']:>6} {m['recall@5']:>6} {m['mrr']:>6}")

    # Soru bazında: hangi yöntem neyi buldu?
    print("\nSoru bazında sıra (doğru maddenin sırası; '-' = ilk 261'de yok):")
    print(f"{'':<50} " + " ".join(f"{n[:9]:>9}" for n in methods))
    for i, q in enumerate(questions):
        row = [results[n]["siralar"][i] for n in methods]
        if any(r is None or r > 1 for r in row):
            print(f"{q['soru'][:50]:<50} " + " ".join(f"{(r or '-'):>9}" for r in row))

    OUT.write_text(json.dumps({"sorular": [q["soru"] for q in questions], **results},
                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
