"""Mevzuat PDF'lerini okuyup her maddeyi ayrı bir chunk olarak kaydeder.

Çıktı: data/chunks.jsonl  (her satır bir madde)
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.jsonl"

SOURCES = {
    "1.5.5464.pdf": "5464 sayılı Banka Kartları ve Kredi Kartları Kanunu",
    "1.5.5411.pdf": "5411 sayılı Bankacılık Kanunu",
}

# "MADDE 12 –", "MADDE 12-", "Geçici Madde 3 –" gibi başlıkları yakalar
MADDE_RE = re.compile(r"^\s*((?:GEÇİCİ|Geçici)\s+)?MADDE\s+(\d+)\s*[–\-—]", re.MULTILINE | re.IGNORECASE)


def pdf_text(path: Path) -> str:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    text = text.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text)


def split_articles(text: str, source: str) -> list[dict]:
    matches = list(MADDE_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        gecici = bool(m.group(1))
        no = m.group(2)
        madde_id = f"{'gecici-' if gecici else ''}{no}"
        # Maddeden önceki satır genelde madde başlığıdır (ör. "Kart çıkarma")
        prev = text[:m.start()].rstrip().rsplit("\n", 1)[-1].strip()
        title = prev if 0 < len(prev) < 80 and not prev.endswith(".") else ""
        chunks.append({
            "id": f"{source.split()[0]}-m{madde_id}",
            "kanun": source,
            "madde": madde_id,
            "baslik": title,
            "text": body,
        })
    return chunks


def main() -> None:
    all_chunks = []
    for fname, source in SOURCES.items():
        chunks = split_articles(pdf_text(RAW / fname), source)
        # Aynı id birden çok çıkarsa (ör. madde metni içinde "MADDE 5 –" referansı) ilkini tut
        seen: set[str] = set()
        for c in chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                all_chunks.append(c)
        print(f"{source}: {len(seen)} madde")
    with OUT.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lens = sorted(len(c["text"]) for c in all_chunks)
    print(f"Toplam {len(all_chunks)} chunk; karakter uzunluğu medyan={lens[len(lens)//2]}, max={lens[-1]}")


if __name__ == "__main__":
    main()
