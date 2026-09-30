"""Kelime bazlı arama (BM25), anlam bazlı arama (dense) ve ikisinin birleşimi (hybrid).

Tüm fonksiyonlar bir soru için 261 maddenin tamamını en alakalıdan en alakasıza sıralı
bir id listesi olarak döndürür; böylece farklı yöntemler aynı şekilde ölçülebilir.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "ve", "veya", "ile", "bu", "bir", "için", "da", "de", "mi", "mı", "mu", "mü", "ne", "gibi",
    "olan", "olarak", "ise", "ki", "her", "daha", "en", "çok", "kadar", "ben", "benim", "bana",
    "sen", "o", "şu", "hangi", "nasıl", "neden", "var", "yok", "mıdır", "midir",
}


def tr_lower(text: str) -> str:
    # Python'un lower()'ı Türkçe I/İ'yi yanlış çevirir: "I".lower() == "i" (olması gereken "ı")
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str, stem: int | None = None) -> list[str]:
    tokens = re.findall(r"\w+", tr_lower(text))
    tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        # Kök kesme (truncation stemming): kelimenin ilk N harfini al.
        # "kartımın", "kartları", "kartın" → "kartı"/"kartl"/... N=5 Türkçe için bilinen iyi bir değer
        tokens = [t[:stem] for t in tokens]
    return tokens


class BM25Search:
    def __init__(self, chunks: list[dict], stem: int | None = None) -> None:
        self.ids = [c["id"] for c in chunks]
        self.stem = stem
        docs = [tokenize(f"{c['baslik']} {c['text']}", stem) for c in chunks]
        self.bm25 = BM25Okapi(docs)

    def rank(self, question: str) -> list[str]:
        scores = self.bm25.get_scores(tokenize(question, self.stem))
        return [self.ids[i] for i in np.argsort(-scores)]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: her listede r. sıradaki belgeye 1/(k+r) puan ver, puanları topla."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=scores.get, reverse=True)


# "54. madde", "madde 54", "54 üncü madde" gibi atıfları yakalar
MADDE_RE = re.compile(r"(?:\bmadde\s*(\d+)\b|\b(\d+)\s*\.?\s*(?:inci|nci|üncü|uncu|ıncı|ncı|ncü|ncu)?\s*madde)", re.IGNORECASE)
KANUN_NO_RE = re.compile(r"\b(5464|5411)\b")


def madde_filter(question: str) -> list[str]:
    """Soru doğrudan bir madde numarası soruyorsa o maddenin id'lerini döndürür."""
    m = MADDE_RE.search(question)
    if not m:
        return []
    no = m.group(1) or m.group(2)
    q = tr_lower(question)
    if (k := KANUN_NO_RE.search(q)):
        laws = [k.group(1)]
    elif "bankacılık kanun" in q:
        laws = ["5411"]
    elif "kart" in q:
        laws = ["5464"]
    else:
        laws = ["5464", "5411"]  # hangi kanun olduğu belli değil: ikisini de öne al
    return [f"{law}-m{no}" for law in laws]


def with_filter(question: str, ranking: list[str]) -> list[str]:
    front = [i for i in madde_filter(question) if i in ranking]
    return front + [i for i in ranking if i not in front]
