"""Farklı embedding modellerini aynı soru setiyle karşılaştırır.

Her model için:
  1. Tüm maddeleri vektöre çevirir
  2. Her soruyu vektöre çevirir, cosine similarity ile en yakın maddeleri bulur
  3. Doğru madde (gold) kaçıncı sırada çıktı → Recall@1, Recall@5, MRR

Kullanım:  python src/embed_compare.py [model_kisa_adi ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
QUESTIONS = ROOT / "eval" / "retrieval_sorular.jsonl"
RESULTS = ROOT / "eval" / "sonuclar"

# e5 modelleri soru/metin için ön ek ister; bge-m3 istemez
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_text(c: dict) -> str:
    # Başlık ve kanun adını metne katmak aramaya bağlam kazandırır
    return f"{c['kanun']} — {c['baslik']}\n{c['text']}"


def evaluate(name: str, chunks: list[dict], questions: list[dict]) -> dict:
    model_id, q_prefix, p_prefix = MODELS[name]
    model = SentenceTransformer(model_id, device="cpu")
    model.max_seq_length = 512

    t0 = time.perf_counter()
    doc_vecs = model.encode([p_prefix + chunk_text(c) for c in chunks],
                            batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_vecs = model.encode([q_prefix + q["soru"] for q in questions], normalize_embeddings=True)
    query_ms = (time.perf_counter() - t0) / len(questions) * 1000

    # Vektörler normalize → dot product = cosine similarity
    scores = q_vecs @ doc_vecs.T
    ids = [c["id"] for c in chunks]

    r1 = r5 = rr = 0.0
    misses = []
    for q, row in zip(questions, scores):
        ranking = [ids[i] for i in np.argsort(-row)]
        rank = next((i + 1 for i, cid in enumerate(ranking) if cid in q["gold"]), None)
        if rank == 1:
            r1 += 1
        if rank and rank <= 5:
            r5 += 1
        rr += 1 / rank if rank else 0
        if not rank or rank > 5:
            misses.append({"soru": q["soru"], "gold": q["gold"], "rank": rank, "top3": ranking[:3]})

    n = len(questions)
    return {
        "model": name,
        "recall@1": round(r1 / n, 3),
        "recall@5": round(r5 / n, 3),
        "mrr": round(rr / n, 3),
        "indeksleme_sn": round(index_s, 1),
        "soru_basina_ms": round(query_ms, 1),
        "kacirilanlar": misses,
    }


def main() -> None:
    chunks = load_jsonl(CHUNKS)
    questions = load_jsonl(QUESTIONS)
    names = sys.argv[1:] or list(MODELS)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"{len(chunks)} madde, {len(questions)} soru\n")
    rows = []
    for name in names:
        res = evaluate(name, chunks, questions)
        (RESULTS / f"embedding_{name}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(res)

    print(f"\n{'model':<10} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'index(s)':>9} {'ms/soru':>8}")
    for r in rows:
        print(f"{r['model']:<10} {r['recall@1']:>6} {r['recall@5']:>6} {r['mrr']:>6} "
              f"{r['indeksleme_sn']:>9} {r['soru_basina_ms']:>8}")


if __name__ == "__main__":
    main()
