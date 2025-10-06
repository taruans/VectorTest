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
import logging
from flask import Flask, render_template, request, redirect
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Loglamayı yapılandır
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
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
            logging.info(f"Yükleniyor: {len(lines)} satır initial_data.txt dosyasından.")
            for line in lines:
                logging.info(f"Vektörleniyor ve ekleniyor: {line}")
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
    logging.info(f"Arama isteği alındı. Sorgu: '{query}'")
    if not query:
        return redirect("/")

    try:
        # Sorgu vektörünü hazırla
        query_vector = vectorizer.encode(query).tolist()

        # Koleksiyonu belleğe yükle
        client.load_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Koleksiyon '{COLLECTION_NAME}' belleğe yüklendi.")

        # Arama işlemi
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=10,
            output_fields=["text", "filename"],
            search_params={"metric_type": "COSINE"}
        )
        logging.info(f"Milvus araması tamamlandı. {len(res[0]) if res else 0} ham sonuç bulundu.")

        # Sonuçları işle, filtrele ve sırala
        results = []
        for hits in res:
            for item in hits:
                raw_score = float(item.distance)
                if 0.0 <= raw_score <= 1.0:
                    similarity = raw_score
                else:
                    similarity = max(0.0, min(1.0, 1.0 - raw_score))

                score_percent = int(round(similarity * 100))

                if score_percent >= 85: label = "🔵 Çok benzer"
                elif score_percent >= 70: label = "🟢 Benzer"
                elif score_percent >= 60: label = "🟡 Kısmen benzer"
                else: label = "⚪ Düşük benzerlik"

                result_data = {
                    "filename": item.entity.get("filename"),
                    "text": item.entity.get("text"),
                    "score": f"{score_percent}%",
                    "score_num": score_percent,
                    "label": label,
                }
                results.append(result_data)
                logging.info(f"İşlenen sonuç: score={score_percent}%, text='{result_data['text'][:100]}...'")

        logging.info(f"Toplam işlenen sonuç sayısı (ham): {len(results)}")

        threshold = 60
        filtered_results = [r for r in results if r.get("score_num", 0) >= threshold]
        logging.info(f"Eşik (>={threshold}%) sonrası sonuç sayısı: {len(filtered_results)}")

        if not filtered_results and results:
            logging.warning("Eşik sonrası sonuç yok. En iyi 3 ham sonuç gösteriliyor (fallback).")
            final_results = sorted(results, key=lambda x: x.get("score_num", 0), reverse=True)[:3]
        else:
            final_results = sorted(filtered_results, key=lambda x: x.get("score_num", 0), reverse=True)

        logging.info(f"Şablona gönderilecek nihai sonuç sayısı: {len(final_results)}")
        return render_template('index.html', results=final_results, query=query)

    except Exception as e:
        logging.error(f"/search rotasında beklenmedik bir hata oluştu: {e}", exc_info=True)
        return render_template('index.html', results=[], query=query, error="Arama sırasında bir hata oluştu.")


if __name__ == "__main__":
    # Flask uygulamasını başlat
    # debug=True geliştirme ortamında otomatik yeniden başlatma sağlar.
    app.run(use_reloader=False)
