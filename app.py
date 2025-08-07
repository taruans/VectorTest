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
from flask import Flask, render_template, request, redirect
from pymilvus import MilvusClient, DataType
from sklearn.feature_extraction.text import HashingVectorizer


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

# Milvus koleksiyon adı ve vektör boyutu
COLLECTION_NAME = "documents"
VECTOR_DIM = 512  # HashingVectorizer için sabit özellik sayısı

# Belgelerin vektörleştirilmesi için HashingVectorizer kullanıyoruz.
# alternate_sign=False seçeneği negatif değerleri engeller ve vektörler
# Milvus'ta saklanabilir hale gelir.
vectorizer = HashingVectorizer(n_features=VECTOR_DIM, alternate_sign=False, norm=None)

# Milvus istemcisini başlatıyoruz. DB_PATH parametresi yerel dosyaya işaret eder.
client = MilvusClient(DB_PATH)

# Koleksiyon mevcut değilse oluşturulur. auto_id=True ayarı ile birincil anahtar
# otomatik olarak oluşturulur. Vektör alanı FLOAT_VECTOR türünde olmalı ve
# boyutu önceden belirlenmelidir.
if not client.has_collection(COLLECTION_NAME):
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=255)

    # İndeks parametreleri hazırlayıp vektör alanına ekliyoruz. AUTOINDEX ve
    # COSINE metriği Milvus Lite tarafından desteklenir.
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
    vector = vectorizer.transform([text]).toarray()[0]
    vector = vector.astype(float).tolist()

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

    return redirect("/")


@app.route("/search", methods=["POST"])
def search():
    """Kullanıcının verdiği sorgu cümlesiyle koleksiyon üzerinde arama yapar."""
    query = request.form.get("query")
    if not query:
        return redirect("/")

    # Sorgu vektörünü hazırla
    query_vector = vectorizer.transform([query]).toarray()[0].astype(float).tolist()

    # Koleksiyonu belleğe yükle
    client.load_collection(collection_name=COLLECTION_NAME)

    # Arama işlemi: en benzer 5 sonucu getirir
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=5,
        output_fields=["text", "filename"],
    )

    # Sonuçları liste haline getir
    results = []
    for hits in res:
        for item in hits:
            results.append(
                {
                    "filename": item.entity.get("filename"),
                    "text": item.entity.get("text"),
                    "score": item.distance,
                }
            )

    return render_template("index.html", results=results)


if __name__ == "__main__":
    # Flask uygulamasını başlat
    # debug=True geliştirme ortamında otomatik yeniden başlatma sağlar.
    app.run(debug=True)