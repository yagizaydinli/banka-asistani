# Banka Asistanı

Türk bankacılık mevzuatı üzerinde, kaynak gösteren soru-cevap asistanı (RAG → Agent).

- Öğrenme notları: [docs/OGRENME_NOTLARI.md](docs/OGRENME_NOTLARI.md)

## Veri
| Dosya | Kaynak |
|---|---|
| `data/raw/1.5.5464.pdf` | 5464 sayılı Banka Kartları ve Kredi Kartları Kanunu — mevzuat.gov.tr |
| `data/raw/1.5.5411.pdf` | 5411 sayılı Bankacılık Kanunu — mevzuat.gov.tr |

## Çalıştırma
```powershell
.\.venv\Scripts\Activate.ps1
python src/ingest.py            # PDF → madde bazlı chunk'lar (data/chunks.jsonl)
python src/embed_compare.py     # embedding modellerini eval setiyle karşılaştır
python src/rag.py "soru"        # kaynak göstererek cevap (Groq; .env'de GROQ_API_KEY gerekir)
python src/eval_answers.py v2   # 49 soruda atıf, reddetme ve sadakat ölçümü
python src/hybrid_compare.py    # dense / BM25 / hybrid / hybrid+filtre retrieval karşılaştırması
```
