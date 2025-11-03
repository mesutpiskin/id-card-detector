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

