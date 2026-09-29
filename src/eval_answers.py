"""RAG cevaplarını ölçer.

Ölçtüklerimiz:
  - Atıf isabeti:   Cevap, doğru maddeyi (gold) kaynak olarak gösterdi mi?
  - Uydurma atıf:   Cevap, kendisine verilmeyen bir madde id'si uydurdu mu?
  - Sadakat (faithfulness): Cevaptaki her bilgi verilen maddelerde var mı? → ikinci bir LLM (hakem) karar verir
  - Doğru reddetme: Cevap maddelerde yoksa (kapsam dışı soru ya da retrieval kaçırdıysa) "cevap yok" dedi mi?

Kullanım:  python src/eval_answers.py <etiket> [--model ollama:qwen3:4b] [--rescore]
           (sonuç: eval/sonuclar/rag_<etiket>.json; hakem her zaman Groq gpt-oss-20b)
"""
import json
import re
import sys
import time
from pathlib import Path

from groq import BadRequestError, RateLimitError

from rag import RAG, build_context, cited_ids, pop_model_arg

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
JUDGE_MODEL = "openai/gpt-oss-20b"  # hakem: cevabı üretenden farklı (daha küçük, ucuz) model
# Modeller "cevap yok" demeyi farklı kelimelerle yapabiliyor (ör. "hiçbir bilgi yoktur", "cevap verilemez");
# sadece istediğimiz kalıbı aramak küçük modelin reddetmelerini kaçırıyordu
REFUSAL_RE = re.compile(
    r"(cevab[ıi]|bilgi)[^.\n]{0,40}\byok(tur)?\b"
    r"|cevap verilemez"
    r"|yer almamaktadır",
    re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))

JUDGE_PROMPT = """Bir cevabın, verilen MADDELER'e sadık olup olmadığını denetliyorsun.
Cevaptaki her iddiayı tek tek kontrol et. Maddelerde açıkça yazmayan bir iddia varsa cevap sadık DEĞİLDİR.
"Verilen maddelerde bu sorunun cevabı yok" gibi bir reddetme her zaman sadıktır.

Sadece şu JSON'u döndür:
{"sadik": true/false, "desteksiz_iddia": "<sadık değilse maddelerde olmayan iddia, sadıksa boş>"}"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge(rag: RAG, question: str, context: str, answer: str) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"MADDELER:\n{context}\n\nSORU: {question}\n\nCEVAP:\n{answer}"},
    ]
    for attempt in range(6):
        try:
            resp = rag.client.chat.completions.create(
                model=JUDGE_MODEL, messages=messages, temperature=0,
                response_format={"type": "json_object"})
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            time.sleep(15 * (attempt + 1))
        except (BadRequestError, json.JSONDecodeError):
            # Hakem geçersiz JSON üretti (Groq bunu 400 hatası olarak döndürür): bir kez daha dene,
            # yine olmazsa bu soruyu "karar yok" say; tüm ölçümü durdurma
            if attempt >= 1:
                return {"sadik": None, "desteksiz_iddia": "hakem geçerli JSON döndürmedi"}
    return {"sadik": None, "desteksiz_iddia": "kota"}


def score(r: dict) -> dict:
    """Kaydedilmiş bir cevabın deterministik metriklerini (yeniden) hesaplar."""
    gold = set(r["gold"])
    atiflar = cited_ids(r["cevap"])
    return {
        **r,
        "atiflar": atiflar,
        "cevaplanabilir": bool(gold & set(r["bulunan"])),  # cevap, LLM'e verilen maddelerde var mı?
        "reddetti": is_refusal(r["cevap"]),
        "atif_isabetli": bool(gold & set(atiflar)),
        "uydurma_atif": sorted(set(atiflar) - set(r["bulunan"])),
    }


def run(model: str, partial: Path) -> list[dict]:
    questions = load_jsonl(EVAL / "retrieval_sorular.jsonl") + load_jsonl(EVAL / "kapsam_disi_sorular.jsonl")
    # Ara kayıt: her cevap anında diske yazılır; ölçüm yarıda kesilirse kaldığı yerden devam eder
    done = {r["soru"]: r for r in load_jsonl(partial)} if partial.exists() else {}
    if done:
        print(f"{len(done)} soru ara kayıttan okundu, kalanlar soruluyor", flush=True)
    rag = RAG(model)
    rows = []
    for i, q in enumerate(questions, 1):
        if q["soru"] in done:
            rows.append(score(done[q["soru"]]))
            continue
        r = rag.answer(q["soru"])
        articles = [a for a in rag.retriever.chunks if a["id"] in r["bulunan"]]
        j = judge(rag, q["soru"], build_context(articles), r["cevap"])
        row = score({**r, "gold": q["gold"], "sadik": j.get("sadik"),
                     "desteksiz_iddia": j.get("desteksiz_iddia", "")})
        rows.append(row)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ok = row["atif_isabetli"] if row["cevaplanabilir"] else row["reddetti"]
        print(f"[{i:>2}/{len(questions)}] {'✓' if ok else '✗'} sadık={row['sadik']} {q['soru'][:70]}", flush=True)
    return rows


def main() -> None:
    # python src/eval_answers.py v1               → soruları LLM'e sorup ölçer
    # python src/eval_answers.py v1 --rescore     → LLM'e sormadan, kayıtlı cevapları yeniden puanlar
    model, args = pop_model_arg(sys.argv[1:])
    label = args[0] if args and not args[0].startswith("--") else "v1"
    out = EVAL / "sonuclar" / f"rag_{label}.json"
    if "--rescore" in sys.argv:
        rows = [score(r) for r in json.loads(out.read_text(encoding="utf-8"))["sorular"]]
    else:
        partial = EVAL / "sonuclar" / f"rag_{label}.partial.jsonl"
        rows = run(model, partial)

    answerable = [r for r in rows if r["cevaplanabilir"]]
    unanswerable = [r for r in rows if not r["cevaplanabilir"]]
    judged = [r for r in rows if r["sadik"] is not None]
    summary = {
        "etiket": label,
        "model": rows[0].get("model", "groq:openai/gpt-oss-120b"),
        "soru_sayisi": len(rows),
        "cevaplanabilir": len(answerable),
        "atif_isabeti": round(sum(r["atif_isabetli"] for r in answerable) / max(len(answerable), 1), 3),
        "gereksiz_reddetme": sum(r["reddetti"] for r in answerable),
        "cevaplanamaz": len(unanswerable),
        "dogru_reddetme": round(sum(r["reddetti"] for r in unanswerable) / max(len(unanswerable), 1), 3),
        "uydurma_atifli_cevap": sum(bool(r["uydurma_atif"]) for r in rows),
        "sadakat": round(sum(r["sadik"] for r in judged) / max(len(judged), 1), 3),
        "ort_sure_sn": round(sum(r["sure_sn"] for r in rows) / len(rows), 2),
        "ort_girdi_token": round(sum(r["girdi_token"] for r in rows) / len(rows)),
    }
    out.write_text(json.dumps({"ozet": summary, "sorular": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    if "--rescore" not in sys.argv:
        partial.unlink(missing_ok=True)  # tam sonuç yazıldı, ara kayda gerek kalmadı
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
