"""
Basit bir Flask uygulaması ile Milvus Lite kullanan bir vektör arama örneği.

Bu uygulama, kullanıcıların metin belgelerini veya doğrudan yazdıkları
metinleri yükleyebilmelerine ve ardından bu metinler arasında arama
yapabilmelerine olanak tanır. Belgeler, scikit‑learn'in HashingVectorizer
kullanılarak sabit boyutlu vektörlere dönüştürülür ve Milvus Lite
veritabanında saklanır.

Not: Milvus Lite, yerel bir veritabanı dosyası üzerinden çalışır. Bu dosya
oluşturulduğunda Milvus arka planda otomatik olarak ayağa kalkar ve
Flask uygulamasıyla aynı işlem içerisinde çalışır. Eğer dinamik kütüphane
yollarıyla ilgili bir sorun yaşanırsa, aşağıdaki kodda yer alan
LD_LIBRARY_PATH ayarı bu durumu düzeltmeye yardımcı olabilir.
"""

import os
import site
import re
import unicodedata
from flask import Flask, render_template, request, redirect
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
# --- Turkish-aware normalization & light stemming for better keyword overlap ---
TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "I": "i",
    "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o",
    "ç": "c", "Ç": "c",
})

def _normalize_text(s: str) -> str:
    s = (s or "").lower().translate(TR_MAP)
    return s

def _tokenize_words(s: str):
    return re.findall(r"\w+", _normalize_text(s))

# very light stemmer for common Turkish suffixes; heuristic but effective for search
_SUFFIXES = [
    "leri", "lari", "nin", "nın", "nun", "nün",
    "dan", "den", "tan", "ten",
    "lar", "ler",
    "dır", "dir", "dur", "dür",
    "tir", "tır", "tur", "tür",
    "da", "de", "ta", "te",
    "ya", "ye", "na", "ne",
    "yı", "yi", "yu", "yü",
    "in", "ın", "un", "ün",
    "a", "e", "i", "ı", "u", "ü",
]

def _stem(token: str) -> str:
    t = token
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t

def norm_tokens(s: str):
    toks = [_stem(t) for t in _tokenize_words(s)]
    return {t for t in toks if len(t) > 2}

# minimal Turkish synonym/related-term expansion for short queries
SYNONYMS = {
    "doktor": {"doktor", "hekim", "dr"},
    "hekim": {"doktor", "hekim"},
    "disci": {"dis", "hekimi", "dishekimi"},
    "psikiyatr": {"psikiyatr", "psikiyatrist", "psikoloji", "psikolojik", "ruhsagligi", "mental"},
    "psikiyatrist": {"psikiyatr", "psikiyatrist", "psikoloji", "psikolojik", "ruhsagligi", "mental"},
    "cerrah": {"cerrah", "cerrahi", "ameliyat", "operasyon"},
}

def expand_query_terms(q_tokens):
    expanded = set(q_tokens)
    for t in q_tokens:
        expanded |= { _stem(x) for x in SYNONYMS.get(t, set()) }
    return expanded


# -----------------------------------------------------------------------------
# Milvus Lite için dinamik kütüphane yolu ayarı
# Milvus Lite çalıştırıldığında libknowhere.so gibi paylaşılan kütüphaneleri
# bulamama hataları yaşanabilir. Bu durumun önüne geçmek için
# site-packages içindeki milvus_lite/lib klasörünü LD_LIBRARY_PATH değişkenine
# ekliyoruz. Bu ayar, uygulama MacOS üzerinde çalıştırıldığında gerek
# kalmayabilir ancak Linux tabanlı ortamlar için faydalıdır.
lib_dir = None
for p in site.getsitepackages():
    candidate = os.path.join(p, "milvus_lite", "lib")
    if os.path.exists(candidate):
        lib_dir = candidate
        break
if lib_dir:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = lib_dir
    if current:
        new_paths = f"{lib_dir}:{current}"
    os.environ["LD_LIBRARY_PATH"] = new_paths

# -----------------------------------------------------------------------------
# Flask ve Milvus yapılandırması

app = Flask(__name__)

# Proje kök dizini ve veritabanı dosyasının tanımlanması
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vector_db.db")

COLLECTION_NAME = "documents"
VECTOR_DIM = 768  # SentenceTransformer "paraphrase-multilingual-mpnet-base-v2" için boyut

# Belgelerin vektörleştirilmesi için SentenceTransformer kullanıyoruz.
vectorizer = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Milvus istemcisini başlatıyoruz. DB_PATH parametresi yerel dosyaya işaret eder.
client = MilvusClient(DB_PATH)

# Koleksiyon mevcut değilse oluşturulur. auto_id=True ayarı ile birincil anahtar
# otomatik olarak oluşturulur. Vektör alanı FLOAT_VECTOR türünde olmalı ve
# boyutu önceden belirlenmelidir.
if not client.has_collection(COLLECTION_NAME):
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=255)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # Örnek veri dosyasından yükleme
    example_file = os.path.join(BASE_DIR, "initial_data.txt")
    if os.path.exists(example_file):
        with open(example_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Yükleniyor: {len(lines)} satır initial_data.txt dosyasından.")
            for line in lines:
                print(f"Vektörleniyor ve ekleniyor: {line}")
            data = [
                {
                    "vector": vectorizer.encode(line).tolist(),
                    "text": line,
                    "filename": "initial_data.txt"
                }
                for line in lines
            ]
            client.insert(collection_name=COLLECTION_NAME, data=data)
            client.flush(collection_name=COLLECTION_NAME)


@app.route("/", methods=["GET"])
def index():
    """Ana sayfa: yükleme ve arama formlarını gösterir."""
    return render_template("index.html", results=None)


@app.route("/upload", methods=["POST"])
def upload():
    """Dosya veya metin yükleme işlemlerini gerçekleştirir."""
    file = request.files.get("file")
    text_input = request.form.get("text_input")

    # Yüklenecek metin ve dosya adı değişkenleri
    text = None
    filename = None

    # Öncelikle dosya yüklenmişse onu kullanıyoruz
    if file and file.filename:
        # Dosyayı UTF-8 olarak çöz; hata oluşursa karakterleri yoksay
        content_bytes = file.read()
        try:
            text = content_bytes.decode("utf-8")
        except Exception:
            text = content_bytes.decode("utf-8", errors="ignore")
        filename = file.filename
    # Dosya yoksa metin kutusundaki veri kullanılır
    elif text_input:
        text = text_input
        filename = "ManualInput"

    # Metin yoksa ana sayfaya yönlendir
    if not text:
        return redirect("/")

    # Metni vektörleştirme
    vector = vectorizer.encode(text).tolist()

    # Milvus'a eklemek üzere veri nesnesi
    data = [
        {
            "vector": vector,
            "text": text,
            "filename": filename,
        }
    ]

    # Veriyi koleksiyona ekle
    client.insert(collection_name=COLLECTION_NAME, data=data)
    # Verinin başarıyla yazıldığından emin olmak için flush işlemi
    client.flush(collection_name=COLLECTION_NAME)

    return redirect("/")


@app.route("/search", methods=["POST"])
def search():
    """Kullanıcının verdiği sorgu cümlesiyle koleksiyon üzerinde arama yapar."""
    query = request.form.get("query")
    print(f"Arama sorgusu: {query}")
    if not query:
        return redirect("/")

    # Sorgu vektörünü hazırla
    query_vector = vectorizer.encode(query).tolist()

    # Koleksiyonu belleğe yükle
    client.load_collection(collection_name=COLLECTION_NAME)

    # Arama işlemi: en benzer 10 sonucu getirir
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=10,
        output_fields=["text", "filename"],
        search_params={"metric_type": "COSINE"}
    )


    # Sonuçları liste haline getir, skoru normalize edip etiketle
    results = []
    for hits in res:
        for item in hits:
            raw = float(item.distance)
            # COSINE in newer PyMilvus typically returns similarity in [0,1] (higher is better).
            # Some setups may expose it as a distance (lower is better). Handle both robustly.
            if 0.0 <= raw <= 1.0:
                similarity = raw  # treat as cosine similarity
            else:
                similarity = max(0.0, min(1.0, 1.0 - raw))  # fallback: convert distance -> similarity
            score_percent = int(round(similarity * 100))

            print(
                f"Bulundu: {item.entity.get('text')} -> raw={raw:.4f} sim={similarity:.4f} percent={score_percent}"
            )

            # Initial label based on embedding similarity (will be adjusted after re-rank)
            if score_percent >= 85:
                label = "🔵 Çok benzer"
            elif score_percent >= 70:
                label = "🟢 Benzer"
            elif score_percent >= 60:
                label = "🟡 Kısmen benzer"
            else:
                label = "⚪ Düşük benzerlik"

            results.append(
                {
                    "filename": item.entity.get("filename"),
                    "text": item.entity.get("text"),
                    "score": f"{score_percent}%",
                    "score_num": score_percent,
                    "label": label,
                }
            )

    # --- Re-rank: combine embedding similarity with Turkish-aware keyword overlap ---
    q_tokens = norm_tokens(query or "")
    q_expanded = expand_query_terms(q_tokens)
    print(f"Re-rank: q_tokens={q_tokens} expanded={q_expanded}")

    for r in results:
        t_tokens = norm_tokens(r["text"])  # normalized + stemmed
        overlap_cnt = len(q_expanded & t_tokens)
        overlap_ratio = (overlap_cnt / len(q_expanded)) if q_expanded else 0.0
        overlap_pct = int(round(overlap_ratio * 100))
        r["overlap"] = overlap_pct

        # Blend: 60% embedding score, 40% keyword overlap (better for short queries)
        final_score = 0.6 * r.get("score_num", 0) + 0.4 * overlap_pct
        r["final_score"] = int(round(final_score))

        # Update primary score/label to reflect final ranking
        r["score_num"] = r["final_score"]
        r["score"] = f"{r['final_score']}%"
        if r["final_score"] >= 85:
            r["label"] = "🔵 Çok benzer"
        elif r["final_score"] >= 70:
            r["label"] = "🟢 Benzer"
        elif r["final_score"] >= 60:
            r["label"] = "🟡 Kısmen benzer"
        else:
            r["label"] = "⚪ Düşük benzerlik"

        print(f"Re-rank item: overlap={overlap_pct}% final={r['final_score']}% text={r['text'][:60]}...")

    print(f"Toplam bulunan sonuç sayısı (ham): {len(results)}")
    # Uygulama eşiği: 50. Hiç sonuç kalmazsa, en iyi 5 sonucu eşiğe bakmadan göster.
    threshold = 50
    filtered_results = [r for r in results if r.get("final_score", 0) >= threshold]
    print(f"Eşik sonrası sonuç sayısı (>={threshold}): {len(filtered_results)}")
    if not filtered_results:
        print("Eşik sonrası sonuç yok. En iyi 5 sonucu gösteriliyor (fallback).")
        filtered_results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)[:5]
    results = sorted(filtered_results, key=lambda x: x.get("final_score", 0), reverse=True)

    return render_template('index.html', results=results, query=query)


if __name__ == "__main__":
    # Flask uygulamasını başlat
    # debug=True geliştirme ortamında otomatik yeniden başlatma sağlar.
    app.run(use_reloader=False)
