"""Maddeleri bge-m3 ile vektöre çevirir, diske kaydeder ve soruya en yakın maddeleri döndürür.

İlk çalıştırmada indeks oluşturulur (~11 dk, CPU). Sonraki çalıştırmalar diskteki
data/index/ dosyalarını okur, sadece soruyu vektöre çevirir.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
MODEL_ID = "BAAI/bge-m3"


def chunk_text(c: dict) -> str:
    return f"{c['kanun']} — {c['baslik']}\n{c['text']}"


class Retriever:
    def __init__(self) -> None:
        self.chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.model = SentenceTransformer(MODEL_ID, device="cpu")
        self.model.max_seq_length = 512
        self.vecs = self._load_or_build()

    def _load_or_build(self) -> np.ndarray:
        vec_path = INDEX_DIR / "bge-m3.npy"
        ids_path = INDEX_DIR / "bge-m3.ids.json"
        ids = [c["id"] for c in self.chunks]
        # Chunk listesi değiştiyse eski indeks geçersizdir → yeniden oluştur
        if vec_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) == ids:
            return np.load(vec_path)
        print("İndeks oluşturuluyor (ilk seferde ~11 dk)...")
        vecs = self.model.encode([chunk_text(c) for c in self.chunks], batch_size=8,
                                 normalize_embeddings=True, show_progress_bar=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, vecs)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        return vecs

    def search(self, question: str, k: int = 5) -> list[dict]:
        q = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
