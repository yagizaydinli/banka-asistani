# Öğrenme Notları — Banka Asistanı

Bu doküman proje boyunca öğrendiğimiz her kavramı **sade dille** anlatır.
Her yeni adımda güncellenir. Amaç: bir mülakatta bu kavramları kendi cümlelerinle anlatabilmen.

> Okuma önerisi: Önce "Büyük Resim"i oku, sonra o haftanın bölümüne geç.
> Bilmediğin bir kelime çıkarsa en alttaki **Sözlük**'e bak.

---

## 0. Büyük Resim — Ne inşa ediyoruz?

Bir kullanıcı şunu soracak:

> "Kredi kartımın limitini banka bana sormadan artırabilir mi?"

Sistemimiz:
1. Bankacılık mevzuatında (kanun, yönetmelik) bu soruyla **ilgili maddeleri bulacak**,
2. Bu maddeleri bir **dil modeline (LLM)** verecek,
3. LLM de **sadece bu maddelere dayanarak**, kaynak göstererek cevap yazacak.

Bu yaklaşımın adı **RAG** (Retrieval-Augmented Generation — "Bulup getirerek zenginleştirilmiş üretim").

### Neden LLM'e direkt sormuyoruz?
- LLM'in bilgisi eğitildiği tarihte donar; mevzuat değişir.
- LLM bilmediği şeyi **uydurabilir** (buna *halüsinasyon* denir). Bankacılıkta bu kabul edilemez.
- "Hangi maddeye göre?" sorusuna cevap veremez. RAG ise kaynağı gösterir.

### Basit benzetme
LLM = çok iyi yazan ama ezberden konuşan bir stajyer.
RAG = stajyerin önüne ilgili kanun sayfalarını açıp "sadece buna bakarak cevapla" demek.

Sonraki haftalarda buna şunlar eklenecek:
- **Değerlendirme (eval):** Sistem ne kadar doğru? Sayıyla ölçmek.
- **Agent:** Soru müşterinin kendi işlemlerini de gerektiriyorsa (ör. "geçen ay limitimi aştım mı?"), sistem veritabanına da bakmaya **kendisi karar verecek**.
- **API + canlıya alma:** Başkalarının kullanabileceği bir servis.

---

## Hafta 1 — Metni sayıya çevirmek ve anlamla arama

### 1.1 LLM (Büyük Dil Modeli)
Milyarlarca metinle eğitilmiş, "bir sonraki kelime ne olmalı?" tahminini çok iyi yapan model.
Örnekler: GPT, Claude, Llama, Qwen, Gemma.

- **Parametre sayısı** (ör. "4B" = 4 milyar): Modelin büyüklüğü. Büyük model genelde daha akıllı ama daha çok bellek ister.
- Bizim laptop (4 GB ekran kartı) ~4B'ye kadar modelleri çalıştırabilir. Büyük modeller için ücretsiz API'ler (Groq, Gemini) kullanacağız.

### 1.2 Token
LLM metni kelime kelime değil, **token** denen parçalar halinde okur.
"Bankacılık" kelimesi 2–4 token olabilir. Türkçe, eklerden dolayı İngilizceye göre **daha çok token** harcar → daha pahalı ve bağlam penceresini daha hızlı doldurur.

**Bağlam penceresi (context window):** LLM'e bir seferde verebileceğin maksimum token sayısı. Tüm kanunu veremeyiz; sadece ilgili parçaları vermemiz gerekir → RAG'in var olma sebebi.

### 1.3 Embedding (Gömme / Vektörleştirme)
Bir metni, **anlamını temsil eden bir sayı listesine** çevirmek.

```
"kredi kartı limiti"      → [0.12, -0.40, 0.88, ... ]  (ör. 768 sayı)
"kartın harcama sınırı"   → [0.10, -0.38, 0.85, ... ]  ← benzer sayılar!
"hava bugün güzel"        → [-0.70, 0.22, 0.01, ... ]  ← çok farklı
```

Kelimeler farklı olsa bile **anlam yakınsa sayılar da yakın** çıkar. Bu işi yapan küçük modellere **embedding modeli** denir (LLM'den ayrı, çok daha küçük ve hızlı).

Bu projede Türkçe için şunları karşılaştıracağız:
- `multilingual-e5-base` — çok dilli, orta boy, hızlı
- `bge-m3` — çok dilli, daha büyük, genelde daha iyi ama yavaş

### 1.4 Cosine Similarity (Kosinüs Benzerliği)
İki vektörün ne kadar **aynı yöne baktığını** ölçer.
- `1.0` → neredeyse aynı anlam
- `0` → alakasız
- Negatif → zıt (pratikte nadir)

Benzetme: Her metin uzayda bir ok. İki ok aynı yöne bakıyorsa anlamları benzer.
Arama = "sorunun okuna en çok benzeyen okları bul".

### 1.5 Chunking (Parçalama)
Kanun metni uzun. Tamamını tek bir vektöre çevirirsek anlam "ortalanır" ve kaybolur.
Bu yüzden metni **parçalara (chunk)** böleriz, her parçanın ayrı embedding'i olur.

Yöntemler:
- **Sabit boyut:** Her 500 karakterde bir kes. Basit ama maddeyi ortasından bölebilir.
- **Yapıya göre:** Mevzuatta doğal bölüm var → **her madde bir chunk**. Bizim domain için en mantıklısı.
- **Örtüşme (overlap):** Parçalar arasında biraz tekrar bırakmak, kesilen cümleler kaybolmasın diye.

Mülakat notu: "Chunk boyutunu nasıl seçtin?" sorusunun iyi cevabı → "Farklı boyutları eval setimle ölçtüm, şu çıktı."

### 1.6 Vektör Veritabanı
Milyonlarca vektör içinde "en benzer 5 tanesini" **hızlıca** bulmak için özel veritabanı.
Örnekler: ChromaDB (basit, başlangıç için), Qdrant, FAISS.

Hız için **indeks** kullanırlar (ör. **HNSW**): Her vektörü tek tek karşılaştırmak yerine, "komşuluk grafiği" üzerinde gezinerek yaklaşık ama çok hızlı arama yapar. Bizim veri küçük olduğu için şimdilik fark hissetmeyeceğiz; ama kavramı bilmek önemli.

### 1.7 Retrieval Eval — "Doğru maddeyi buldu mu?"
LLM'e geçmeden önce **arama kısmını** ölçeriz:
- Bir soru listesi hazırlarız, her soru için **doğru cevabın hangi maddede olduğunu** not ederiz.
- **Recall@5:** Doğru madde, ilk 5 sonuç içinde çıktı mı? (Kaç soruda çıktı / toplam soru)
- **MRR:** Doğru madde kaçıncı sırada? 1. sıradaysa 1 puan, 2.'deyse 0.5, 3.'teyse 0.33...

Bu sayılar sayesinde "e5 mi bge-m3 mü daha iyi?" sorusuna **fikirle değil ölçümle** cevap veririz.

### 1.8 İlk deney: Hangi embedding modeli? (sonuçlar)

**Kurulum:** 2 kanun → 261 madde (her madde bir chunk). 44 soruluk test seti; sorular
müşteri ağzından yazıldı ("Kartım çalındı, ne kadarından sorumluyum?"), kanun diliyle değil.
Script: `src/embed_compare.py` — ayrıntılı sonuçlar `eval/sonuclar/` içinde.

| Model | Recall@1 | Recall@5 | MRR | 261 maddeyi indeksleme | Soru başına |
|---|---|---|---|---|---|
| e5-small | 0.64 | 0.77 | 0.70 | 61 sn | 17 ms |
| e5-base | 0.64 | **0.84** | 0.72 | 183 sn | 54 ms |
| bge-m3 | **0.68** | **0.89** | **0.78** | 645 sn (~11 dk) | 165 ms |

**Nasıl okunur?** bge-m3, 44 sorunun 39'unda doğru maddeyi ilk 5 sonuç içinde buldu; e5-base 37'sinde.
En iyi model bge-m3 ama e5-base'den **3,5 kat, e5-small'dan 10 kat yavaş** → gerçek hayatta hep bu takas yapılır.

**Karar:** Kalite önemli ve indeksleme bir kere yapılıyor → **bge-m3** ile devam.
Soru başına 165 ms kabul edilebilir (LLM cevabı zaten saniyeler sürecek).
bge-m3 "kelime uçurumu" sorularında da daha iyi: "devlet güvencesi" sorusunda doğru maddeyi 208. sıradan 11. sıraya taşıdı.
Ama madde numarası sorularını o da kaçırıyor → sorun modelde değil, **yöntemde** (dense arama).

### 1.9 Kaçırılan sorulardan öğrendiklerimiz (en değerli kısım)

Skordan çok, **neyi neden kaçırdığı** önemli. e5-base'in kaçırdıkları 3 türe ayrılıyor:

**1. Numara/anahtar kelime soruları**
- "5464 sayılı kanunun 22. maddesi ne düzenliyor?" → doğru madde 16. sırada
- "Bankacılık Kanunu 54. madde neyi düzenler?" → 108. sırada!

Neden? Embedding **anlamı** yakalar, "22" ile "54" sayısının anlamsal farkı yoktur.
Çözüm (Hafta 2): **Hybrid search** — kelime eşleştirmesi yapan BM25'i de ekle,
ya da soruda madde numarası varsa doğrudan filtrele.

**2. Kelime uçurumu (vocabulary gap)**
- "Bankadaki param devlet güvencesinde mi?" → doğru madde (Tasarruf Mevduatı **Sigorta** Fonu) 208. sırada
- "Hesabıma on yıl dokunmazsam ne olur?" → doğru madde (**Zamanaşımı**) 25. sırada

Müşteri "güvence" diyor, kanun "sigorta" diyor. Müşteri "dokunmazsam" diyor, kanun "zamanaşımı".
Çözüm (Hafta 2): **Query rewriting** — LLM'e soruyu önce "kanun diline" çevirtmek.

**3. Uzun chunk'ta anlam kaybolması**
- "Kredi kartımı iptal ettirmek istiyorum" → cevap 25. maddenin **son cümlesinde**; madde asıl olarak faiz artışını anlatıyor, bu cümle ortalamada kayboluyor.
- "Kart hamili ne demek?" → Tanımlar maddesi 3000 karakter, 20 farklı tanım içeriyor.

Çözüm (Hafta 2): **Fıkra bazlı chunking** veya küçük parça ile ara, büyük parçayı LLM'e ver (*parent-document retrieval*).

> 💡 Mülakat cümlesi: "Baseline'da Recall@5 0.84 çıktı. Hataları inceleyince üç kategori
> buldum: numara referansları, kelime uçurumu ve uzun chunk'lar. Her biri için ayrı çözüm denedim
> ve etkisini aynı eval setiyle ölçtüm."

---

## Proje Günlüğü

Her adımda ne yaptığımızın kısa kaydı.

| Tarih | Ne yaptık | Sonuç / Öğrendik |
|---|---|---|
| 2026-09-29 | Proje klasörü + Python ortamı kuruldu, 5464 ve 5411 sayılı kanunlar indirildi | PDF'ten madde bazlı 261 chunk |
| 2026-09-29 | 44 soruluk retrieval eval seti yazıldı, 3 embedding modeli karşılaştırıldı | bge-m3 en iyi (Recall@5 = 0.89) ama en yavaş; hatalar 3 türe ayrıldı |
| 2026-09-29 | RAG cevap üretimi (gpt-oss-120b, Groq) + 4 metrikli cevap eval'i (hakem: gpt-oss-20b) | Atıf 0.97, sadakat 0.96, reddetme 0.90; "150→50" ve "kendi ezberinden cevap" hataları. Ayrıntı: docs/ogrenme.html §9–§11 |

---

## Sözlük

| Terim | Sade anlamı |
|---|---|
| **LLM** | Metin üreten büyük dil modeli |
| **Token** | LLM'in okuduğu metin parçası (kelimeden küçük olabilir) |
| **Context window** | LLM'e tek seferde verilebilecek metin miktarı |
| **Embedding** | Metnin anlamını temsil eden sayı listesi |
| **Vektör** | Sayı listesi (embedding'in matematiksel adı) |
| **Cosine similarity** | İki vektörün anlam benzerliği (0–1) |
| **Chunk** | Metnin aranabilir küçük parçası |
| **Vektör DB** | Vektörleri saklayıp benzerlik araması yapan veritabanı |
| **HNSW** | Vektör DB'lerin hızlı arama için kullandığı indeks yapısı |
| **RAG** | İlgili metni bulup LLM'e vererek cevap ürettirme |
| **Halüsinasyon** | LLM'in gerçek olmayan bilgiyi uydurması |
| **Recall@k** | Doğru sonuç ilk k sonuç içinde mi? |
| **MRR** | Doğru sonucun sıralamadaki yerine göre puan |
| **Eval** | Sistemin başarımını sayılarla ölçme |
