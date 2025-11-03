# Kimlik Kartı Tespiti

[English](README.md) | Türkçe

Bu proje TensorFlow 2 (SavedModel) ve opsiyonel OCR (EasyOCR) ile kimlik kartı tespiti yapar. Eski TF1 bağımlılıkları kaldırıldı; `object_detection.*` importları artık yok.

## Modernizasyon Notları (2025)

- TF1 grafik ve TF Object Detection API bağımlılığı kaldırıldı.
- `model/saved_model/` içindeki TF2 SavedModel doğrudan yükleniyor; `serving_default` imzası otomatik bulunuyor.
- `id_card_detection_image.py` komut satırı argümanları:
  - `--image`: Görsel yolu (mutlak/göreli)
  - `--min_score`: Skor eşiği (0–1, varsayılan 0.60)
  - `--ocr`: Kırpılan ROI üzerinde OCR çalıştırır (EasyOCR)
- `data/labelmap.pbtxt` için hafif bir label map okuyucu eklendi.
- Görselleştirme OpenCV ile yapılır; en yüksek skorlu kutu yeşil, diğerleri kırmızı çizilir. ROI `output_cropped.png` olarak kaydedilir.

> Not: SavedModel çıktı anahtarları modele göre değişebilir (`detection_boxes`, `detection_scores`, `detection_classes` varsayılan kabul edilir).

## Kurulum ve Çalıştırma (Önerilen TF2 + OCR)

Önce TensorFlow’u platformunuza göre kurun, sonra diğer paketleri yükleyin.

### 1) TensorFlow kurulumu

- macOS (Apple Silicon, M-serisi):
```bash
pip3 install tensorflow-macos==2.16.1 tensorflow-metal==1.2.0
```

- macOS (Intel) veya Linux (CPU):
```bash
pip3 install tensorflow==2.20.0
```

- Windows (CPU):
```bash
pip3 install tensorflow==2.17.1
```

> İpucu: Python 3.12 için `python3 -m pip install --upgrade pip` çalıştırın.

### 2) Diğer bağımlılıklar
```bash
pip3 install -r requirements-modern.txt
```

### 3) Çalıştırma
```bash
python3 id_card_detection_image.py --image /mutlak/veya/goreli/yol.jpg --ocr --min_score 0.6
```

- Windows örnek yolu: `C:\\path\\to\\image.jpg`
- Kırpılan ROI proje kökünde `output_cropped.png` olarak kaydedilir.
- `--ocr` ile metinler terminale yazdırılır.

### YOLO entegrasyonu (opsiyonel)

YOLO arka ucunu kur:
```bash
pip3 install ultralytics
```

TF2 yerine YOLO ile (görsel):
```bash
python3 id_card_detection_image.py --image /yol/gorsel.jpg --yolo_model yolov8n.pt --min_score 0.4 --ocr
```

YOLO ile kamera ve OCR snapshot paneli (1–9 ile seçip OCR):
```bash
python3 id_card_detection_camera.py --yolo_model yolov8n.pt --min_score 0.4 --ocr
```

Kamera kısayolları: q çıkış, p duraklat/devam, s kamera durdur, b başlat, 1–9 seçili kırpımda OCR.

## Komut Satırı Parametreleri

Görsel scripti (`id_card_detection_image.py`):

- `--image` (string): Görsel dosya yolu (mutlak/göreli). Verilmezse `test_images/image1.png` kullanılır.
- `--min_score` (float, varsayılan 0.60): [0,1] aralığında güven skoru eşiği. Kutuların çizilmesi ve kırpılacak ROI için kullanılır. Daha fazla aday görmek için 0.3–0.5 aralığı denenebilir.
- `--ocr` (bayrak): Verilirse kırpılmış ROI üzerinde EasyOCR çalıştırılır ve çıkan metin satırları terminale yazdırılır.
- `--yolo_model` (string, opsiyonel): Verilirse dedektör arka ucu YOLO’ya alınır. Yerel `.pt` yolu veya `yolov8n.pt` gibi bir model adı kabul eder. Verilmezse `model/saved_model/` altındaki TF2 SavedModel kullanılır.

Kamera scripti (`id_card_detection_camera.py`):

- `--camera` (int, varsayılan 0): Kamera cihaz indeksi. Birden fazla kamera varsa 1 veya 2’yi deneyin.
- `--min_score` (float, varsayılan 0.50): Tespitlerin çizimi ve snapshot listelenmesi için skor eşiği.
- `--yolo_model` (string, opsiyonel): YOLO model yolu/adı; görsel scripttekiyle aynı davranır.
- `--ocr` (bayrak): OCR iş akışını etkinleştirir. Etkin olduğunda:
  - Sağ panelde skora göre sıralı en fazla 9 kırpım gösterilir.
  - Klavyeden 1–9’a basıldığında ilgili kırpımda OCR çalışır; sonuçlar sağ panelde “OCR Result” altında ve terminalde gösterilir.

Pencere kısayolları (kamera):

- `q`: Çıkış.
- `p`: Duraklat/Devam (duraklatıldığında son kare sabit kalır).
- `s`: Kamerayı durdur (release).
- `b`: Kamerayı başlat/yeniden başlat.
- `1–9`: Sağ paneldeki N’inci snapshot’ta OCR çalıştır (sadece `--ocr` verildiyse).

Çıktılar:

- Görsel scripti: Görselde kutular çizilir ve en iyi ROI `output_cropped.png` olarak kaydedilir. OCR metni terminale yazdırılır.
- Kamera scripti: Canlı akış üzerinde kutular çizilir; sağ panelde küçük önizlemeler ve (etkinse) OCR sonuçları yer alır.

## Alternatif: Eski TF1 Akışı

Python 3.7 ve TensorFlow 1.15 gerektirir. Modern sistemlerde kurulumu zordur.

```bash
pip3 install -r requirements.txt
python3 id_card_detection_image.py
```

> Uyarı: TF1 bağımlılıkları (özellikle macOS/ARM) güncel Python sürümleriyle uyumlu değildir.

## Platform Notları

- macOS Apple Silicon: `tensorflow-macos` + `tensorflow-metal` gerekir. İlk çalıştırmada küçük gecikmeler normaldir.
- macOS Intel / Linux: CPU `tensorflow` yeterlidir.
- Windows: Gerekirse Visual C++ Redistributable kurulu olmalıdır.

## Örnek Komutlar

- macOS Apple Silicon:
```bash
pip3 install tensorflow-macos==2.16.1 tensorflow-metal==1.2.0
pip3 install -r requirements-modern.txt
python3 id_card_detection_image.py --image /Users/siz/Downloads/test_data.jpg --ocr --min_score 0.6
```

- Linux / macOS Intel:
```bash
pip3 install tensorflow==2.20.0
pip3 install -r requirements-modern.txt
python3 id_card_detection_image.py --image ./test_images/image1.png --min_score 0.6
```

