"""RAG: soruya en yakın maddeleri bul, LLM'e ver, kaynak göstererek cevaplat.

Kullanım:  python src/rag.py "Kartım çalındı, ne kadarından sorumluyum?"
"""
import re
import sys
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from retriever import Retriever

load_dotenv()

MODEL = "openai/gpt-oss-120b"
TOP_K = 5
MAX_CHARS_PER_ARTICLE = 2500  # çok uzun maddeleri kırp → token (ve kota) tasarrufu

SYSTEM_PROMPT = """Sen Türk bankacılık mevzuatı konusunda bilgi veren bir asistansın.

Kurallar:
1. SADECE sana verilen MADDELER'e dayanarak cevap ver. Kendi bilgini ekleme.
2. Her bilginin sonunda kaynağını köşeli parantezle yaz, ör. [5464-m9]. Yalnızca verilen id'leri kullan.
3. Maddelerde cevap yoksa "Verilen maddelerde bu sorunun cevabı yok." de ve başka bir şey yazma.
4. Sade, günlük Türkçe kullan; müşteriye anlatır gibi kısa cevap ver (en fazla 5 cümle)."""


def build_context(articles: list[dict]) -> str:
    parts = []
    for a in articles:
        text = a["text"][:MAX_CHARS_PER_ARTICLE]
        parts.append(f"<madde id=\"{a['id']}\" kanun=\"{a['kanun']}\" baslik=\"{a['baslik']}\">\n{text}\n</madde>")
    return "\n\n".join(parts)


def ask_llm(client: Groq, question: str, context: str) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"MADDELER:\n{context}\n\nSORU: {question}"},
    ]
    for attempt in range(6):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
            usage = {
                "sure_sn": round(time.perf_counter() - t0, 2),
                "girdi_token": resp.usage.prompt_tokens,
                "cikti_token": resp.usage.completion_tokens,
            }
            return resp.choices[0].message.content.strip(), usage
        except RateLimitError:
            # Ücretsiz katmanda dakikalık kota dolunca bekleyip tekrar dene
            wait = 15 * (attempt + 1)
            print(f"  kota doldu, {wait} sn bekleniyor...")
            time.sleep(wait)
    raise RuntimeError("Groq kotası aşıldı, sonra tekrar deneyin")


def cited_ids(answer: str) -> list[str]:
    # gpt-oss bazen [..] yerine 【..】 kullanıyor; ikisini de tanı
    return re.findall(r"[\[【](\d{4}-m[\w-]+)[\]】]", answer)


class RAG:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.client = Groq(timeout=60)

    def answer(self, question: str) -> dict:
        articles = self.retriever.search(question, k=TOP_K)
        text, usage = ask_llm(self.client, question, build_context(articles))
        return {
            "soru": question,
            "cevap": text,
            "bulunan": [a["id"] for a in articles],
            "atiflar": cited_ids(text),
            **usage,
        }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Bankam ben istemeden kredi kartı limitimi artırabilir mi?"
    r = RAG().answer(question)
    print(f"\nSORU: {r['soru']}\n")
    print(r["cevap"])
    print(f"\nBulunan maddeler: {r['bulunan']}")
    print(f"Süre: {r['sure_sn']} sn · token: {r['girdi_token']} girdi + {r['cikti_token']} çıktı")
