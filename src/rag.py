"""RAG: soruya en yakın maddeleri bul, LLM'e ver, kaynak göstererek cevaplat.

Kullanım:
  python src/rag.py "Kartım çalındı, ne kadarından sorumluyum?"
  python src/rag.py --model ollama:qwen3:4b-instruct "Kartım çalındı, ne kadarından sorumluyum?"
  python src/rag.py --retrieval dense "..."        → Hafta 1 davranışı (sadece embedding)

Model adı "<servis>:<model>" biçiminde:
  groq:openai/gpt-oss-120b   → Groq bulut API (varsayılan)
  ollama:qwen3:4b            → bilgisayardaki Ollama (http://localhost:11434)
"""
import re
import sys
import time

import requests
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from retriever import Retriever

load_dotenv()

DEFAULT_MODEL = "groq:openai/gpt-oss-120b"
DEFAULT_RETRIEVAL = "hybrid"  # Hafta 2: dense + BM25 + madde filtresi (eski davranış: --retrieval dense)
OLLAMA_URL = "http://localhost:11434/api/chat"
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


def ask_groq(client: Groq, model: str, messages: list[dict]) -> tuple[str, dict]:
    for attempt in range(6):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
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


def ask_ollama(model: str, messages: list[dict]) -> tuple[str, dict]:
    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, timeout=600, json={
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # qwen3 gibi "düşünen" modellerde düşünme adımını kapat → çok daha hızlı
        "options": {
            "temperature": 0,
            "num_ctx": 6144,      # varsayılan bağlam penceresi 5 maddeye yetmeyebilir
            "num_predict": 400,   # cevap en fazla 400 token; kontrolden çıkan üretimi keser
        },
    })
    resp.raise_for_status()
    data = resp.json()
    usage = {
        "sure_sn": round(time.perf_counter() - t0, 2),
        "girdi_token": data.get("prompt_eval_count", 0),
        "cikti_token": data.get("eval_count", 0),
    }
    return data["message"]["content"].strip(), usage


def cited_ids(answer: str) -> list[str]:
    # gpt-oss bazen [..] yerine 【..】 kullanıyor; ikisini de tanı
    return re.findall(r"[\[【](\d{4}-m[\w-]+)[\]】]", answer)


class RAG:
    def __init__(self, model: str = DEFAULT_MODEL, retrieval: str = DEFAULT_RETRIEVAL) -> None:
        self.model = model
        self.retrieval = retrieval  # "dense" veya "hybrid"
        self.provider, self.model_name = model.split(":", 1)
        self.retriever = Retriever()
        self.client = Groq(timeout=60)  # hakem de bunu kullanıyor

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        if self.provider == "groq":
            return ask_groq(self.client, self.model_name, messages)
        if self.provider == "ollama":
            return ask_ollama(self.model_name, messages)
        raise ValueError(f"Bilinmeyen servis: {self.provider}")

    def answer(self, question: str) -> dict:
        articles = self.retriever.search(question, k=TOP_K, mode=self.retrieval)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"MADDELER:\n{build_context(articles)}\n\nSORU: {question}"},
        ]
        text, usage = self.generate(messages)
        return {
            "soru": question,
            "model": self.model,
            "retrieval": self.retrieval,
            "cevap": text,
            "bulunan": [a["id"] for a in articles],
            "atiflar": cited_ids(text),
            **usage,
        }


def pop_arg(argv: list[str], name: str, default: str) -> tuple[str, list[str]]:
    """'--name deger' çiftini argüman listesinden çıkarır."""
    if name in argv:
        i = argv.index(name)
        return argv[i + 1], argv[:i] + argv[i + 2:]
    return default, argv


if __name__ == "__main__":
    model, args = pop_arg(sys.argv[1:], "--model", DEFAULT_MODEL)
    retrieval, args = pop_arg(args, "--retrieval", DEFAULT_RETRIEVAL)
    question = " ".join(args) or "Bankam ben istemeden kredi kartı limitimi artırabilir mi?"
    r = RAG(model, retrieval).answer(question)
    print(f"\nSORU: {r['soru']}  ({r['model']}, {r['retrieval']})\n")
    print(r["cevap"])
    print(f"\nBulunan maddeler: {r['bulunan']}")
    print(f"Süre: {r['sure_sn']} sn · token: {r['girdi_token']} girdi + {r['cikti_token']} çıktı")
