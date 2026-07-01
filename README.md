# Termal Görüntülerde İHA Tespiti ve Kilit Alma Takip Sistemi

YOLOv8, Kalman Filtresi ve Macar Algoritması kullanılarak termal IR görüntülerde küçük insansız hava araçlarının (İHA/drone) tespiti, takibi ve kilit alınması.

PROJENİN README'SİNİ OKUMADAN BU KISMI OKUYUN:
- BEN ELİMDEKİ CİHAZIMLA EN FAZLA NELER YAPABİLİRİM DİYE DENEDİĞİM BİR PROJE OLDU, 6 SAATLİK BİR EĞİTİM SÜRECİ OLDUĞU İÇİN MODEL ÇOK İYİ BİR ŞEKİLDE HEDEFLERİ TESPİT EDEMİYOR. SİZ YAPARKEN EPOCH SAYISINI ARTIRIP İMAGE SİZE'INI DA NORMAL OLARAK TUTARSANIZ MODELİNİZ DAHA KARARLI DAVRANIR AMA ELİNİZDEKİ CİHAZA GÖRE DE EĞİTİM SÜRESİ ARTABİLİR M2 AİR BİR CİHAZ KULLANARAK TÜM VERİNİN YÜZDE 10'U , 6 SAATLİK 20 EPOCH BİR EĞİTİMLE BU ŞEKİLDE BİR SONUÇ ALINIYOR. 

SAYGILARIMLA  
MEHMET DEĞİRMENCİ

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Pipeline](#pipeline)
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Kullanım](#kullanım)
- [Dosya Yapısı](#dosya-yapısı)
- [Modüller](#modüller)
- [Sonuçlar](#sonuçlar)
- [İyileştirme Fırsatları](#iyileştirme-fırsatları)

---

## Proje Özeti

Termal kameralar, normal RGB kameraların göremediği düşük ışık ve gece koşullarında çalışabilir. Drone motoru ısı yaydığı için termal görüntüde parlak görünür — bu tespiti kolaylaştırır. Ancak görüntüler gürültülü ve düşük kontrastlıdır, drone'lar çok küçük olabilir (14x11 piksel).

Bu proje şu problemleri çözer:
- **Tespit:** Her frame'de drone nerede?
- **Takip:** Drone nereden nereye gidiyor?
- **Kilit alma:** Drone kaybolursa tahminle devam et, geri gelince kilitle
- **Değerlendirme:** Sistem ne kadar iyi çalışıyor?

---

## Pipeline

```
Termal Video Frame
      ↓
┌─────────────────────┐
│   YOLOv8 Tespiti    │  → Her frame'de "drone şurada" koordinatı
│   (+ SAHI opsiyonel)│
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Macar Algoritması  │  → "Bu tespit hangi track'e ait?"
│  (association.py)   │    Maliyet matrisi + optimal eşleştirme
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Kalman Filtresi    │  → Koordinatları yumuşat
│  (kalman.py)        │    YOLO kaçırırsa tahminle devam et
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Durum Makinesi     │  → ARANIYOR → KİLİTLİ ↔ TAHMİN → KAYIP
│  (tracker.py)       │
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Görselleştirme     │  → Bbox, ID, durum etiketi çiz
│  (visualize.py)     │
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Değerlendirme      │  → MOTA, IDF1, FPS
│  (evaluate.py)      │
└─────────────────────┘
```

---

## Kurulum

### Gereksinimler
- Python 3.11
- Apple Silicon Mac (MPS) veya CUDA destekli GPU

### Paketler

```bash
pip install ultralytics sahi opencv-python filterpy scipy motmetrics
```

Kurulu paketleri doğrula:

```bash
pip show ultralytics sahi opencv-python filterpy scipy motmetrics
```

> **Not:** `python3.11` kullan. `python3` komutu farklı bir versiyona işaret edebilir.

---

## Veri Seti

[Anti-UAV410](https://github.com/ZhaoJ9014/Anti-UAV) veri seti kullanılmıştır.

### Yapı
```
dataset/
  train/    (200 sequence)
  val/      (90 sequence)
  test/     (120 sequence)
```

Her sequence klasöründe:
- Sıralı `.jpg` frame'ler (640x512, IR termal)
- `IR_label.json` → ground truth etiketler

### JSON Format
```json
{
  "exist": [1, 1, 0, ...],
  "gt_rect": [[x, y, w, h], [x, y, w, h], null, ...]
}
```
- `exist[i] = 1` → i. frame'de drone görünür
- `gt_rect[i] = [x, y, w, h]` → sol-üst köşe + genişlik/yükseklik (piksel)

---

## Kullanım

### 1. Veri Setini YOLO Formatına Dönüştür

```bash
python3.11 convert_to_yolo.py
```

Tüm dataset (209K frame) için. Küçük subset için:

```bash
python3.11 create_small_dataset.py
```

20 train / 10 val / 10 test sequence kullanır.

### 2. Modeli Eğit

```bash
python3.11 train.py
```

`train.py` ayarları:
```python
model.train(
    data="yolo_dataset_small/data.yaml",
    epochs=20,
    imgsz=320,      # 640 daha iyi ama yavaş
    batch=16,
    device="mps",   # Apple Silicon için, CUDA için "cuda"
    patience=10,
    save_period=5,
)
```

Eğitim çıktısı: `runs/uav_detector-N/weights/best.pt`

### 3. Model Testi

`test_model.ipynb` notebook'unu aç:

```python
from ultralytics import YOLO

model = YOLO("runs/uav_detector-5/weights/best.pt")
results = model("görüntü.jpg", conf=0.3)

# Bbox koordinatları
boxes = results[0].boxes.xyxy   # (x1, y1, x2, y2)
confs = results[0].boxes.conf   # güven skorları
```

### 4. Tam Pipeline — Tracking

```python
from ultralytics import YOLO
from src.tracker import Tracker
from src.visualize import draw_tracks
import cv2

model   = YOLO("runs/uav_detector-5/weights/best.pt")
tracker = Tracker(max_missing=3)

for frame_path in frame_listesi:
    img = cv2.imread(frame_path)

    # Tespit
    results = model(frame_path, conf=0.3, verbose=False)
    detections = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        detections.append(((x1+x2)/2, (y1+y2)/2))

    # Track güncelle
    tracks = tracker.update(detections)

    # Çiz
    img = draw_tracks(img, tracks)
    cv2.imshow("demo", img)
```

### 5. SAHI ile Gelişmiş Tespit

```python
from src.detector import load_model, detect

sahi_model = load_model("runs/uav_detector-5/weights/best.pt", conf=0.3)
detections = detect(sahi_model, "görüntü.jpg", slice_size=256, overlap=0.2)
# → [(cx, cy, conf), ...]
```

### 6. Demo Videosu Oluştur

`test_model.ipynb`'deki demo hücresini çalıştır → `runs/demo.mp4`

### 7. Değerlendirme

```python
from src.evaluate import evaluate
from src.tracker import Tracker

sequences = [
    ("dataset/test/seq_001", "dataset/test/seq_001/IR_label.json"),
    ...
]

tracker = Tracker(max_missing=3)
summary = evaluate(tracker, model, sequences, conf=0.3)
```

---

## Dosya Yapısı

```
.
├── src/
│   ├── kalman.py              # Kalman filtresi
│   ├── association.py         # Macar algoritması
│   ├── tracker.py             # Durum makinesi
│   ├── visualize.py           # Görselleştirme
│   ├── detector.py            # SAHI + YOLO wrapper
│   ├── evaluate.py            # MOTA, IDF1, FPS
│   ├── association_aciklama.md
│   ├── tracker_aciklama.md
│   └── visualize_aciklama.md
├── dataset/                   # Anti-UAV410 ham veri (git ignore)
├── yolo_dataset/              # Tüm veri YOLO formatında (git ignore)
├── yolo_dataset_small/        # 20 sequence subset (git ignore)
├── runs/                      # Eğitim çıktıları (git ignore)
│   ├── uav_detector-5/
│   │   └── weights/
│   │       ├── best.pt        # En iyi model (Epoch 17)
│   │       └── last.pt
│   └── demo.mp4               # Demo videosu
├── convert_to_yolo.py         # Dataset dönüştürme
├── create_small_dataset.py    # Küçük subset oluşturma
├── train.py                   # YOLOv8 eğitim scripti
├── test_model.ipynb           # Test, görselleştirme, demo
├── colab_train.ipynb          # Google Colab versiyonu
├── proje_anlatis.md           # Sıfırdan detaylı anlatım
├── session_ozeti.md           # Teknik özet
└── .gitignore
```

---

## Modüller

### `src/kalman.py` — Kalman Filtresi

Drone'un konumunu ve hızını tahmin eder. YOLO gürültülü koordinat verdiğinde yumuşatır, drone kaybolduğunda tahminle devam eder.

**State vektörü:** `[x, y, vx, vy]` — konum + hız

```python
tracker = kalmanFilter(x=100, y=200)
pred    = tracker.predict()        # → tahmini konum
updated = tracker.update(x, y)    # → YOLO ölçümü ile güncelle
```

**Matrisler:**
| Matris | Boyut | Açıklama |
|---|---|---|
| F | 4x4 | Transition: `x_yeni = x + vx` |
| H | 2x4 | Measurement: state'ten [x,y] seç |
| P | 4x4 | State kovaryansı (belirsizlik) |
| R | 2x2 | YOLO gürültüsü = 10 |
| Q | 4x4 | Model gürültüsü = 0.1 |

**Kalman Gain:** R büyük → tahminine güven. R küçük → YOLO'ya güven.

---

### `src/association.py` — Macar Algoritması

N tracker ve M tespit arasında optimal eşleştirme yapar.

```python
matches, unmatched = associate(
    trackers=[(100,200), (400,300)],
    detections=[(102,198), (405,297), (250,250)],
    max_dist=50
)
# matches   = [(0,0), (1,1)]  → T0→D0, T1→D1
# unmatched = [2]             → false positive, at
```

**Adımlar:**
1. Maliyet matrisi: her tracker-tespit arası piksel mesafesi
2. `scipy.linear_sum_assignment` ile toplam maliyeti minimize et
3. `max_dist` filtresi: 50 pikselden uzak eşleştirmeleri reddet

---

### `src/tracker.py` — Durum Makinesi

```
ARANIYOR → drone bulununca → KİLİTLİ
KİLİTLİ → drone kaybolunca → TAHMİN
TAHMİN  → drone tekrar bulununca → KİLİTLİ
TAHMİN  → missing_count > max_missing → KAYIP → sil
```

```python
tracker = Tracker(max_missing=3)
tracks  = tracker.update(detections)  # her frame'de çağır

for t in tracks:
    print(t.id, t.state.name)  # → "0 KİLİTLİ"
```

---

### `src/visualize.py` — Görselleştirme

```python
frame = draw_tracks(frame, tracks)
# KİLİTLİ  → yeşil bbox
# TAHMİN   → turuncu bbox
# ARANIYOR → mavi bbox
```

Renk kodları (BGR formatı — OpenCV):
```python
COLORS = {
    "KILITLI":  (0, 255, 0),    # Yeşil
    "TAHMIN":   (0, 165, 255),  # Turuncu
    "ARANIYOR": (255, 0, 0),    # Mavi
    "KAYIP":    (0, 0, 255),    # Kırmızı
}
```

---

### `src/detector.py` — SAHI Wrapper

Görüntüyü slice'lara bölerek küçük drone'ların tespitini iyileştirir.

**Problem:** 14x11 piksellik drone, 320x320 resize sonrası daha da küçülür.
**Çözüm:** 256x256'lık parçalarda tara → her parçada drone görece büyür.

```python
model = load_model("best.pt", conf=0.3)
dets  = detect(model, "frame.jpg", slice_size=256, overlap=0.2)
# overlap=0.2 → slice sınırındaki drone'lar kaçırılmaz
```

---

### `src/evaluate.py` — Değerlendirme

```python
summary = evaluate(tracker, model, sequences, conf=0.3)
```

**Metrikler:**
- **MOTA** = `1 - (FP + FN + ID_switch) / toplam_gt` → 1'e yakın iyi
- **IDF1** → doğru ID eşleşme oranı → 1'e yakın iyi
- **FPS** → saniyede kaç frame

---
## Demo video


https://github.com/user-attachments/assets/c00101f3-16ff-4445-bd67-b056b9e8b7e4



## Sonuçlar

### Model Performansı (best.pt — Epoch 17)
| Metrik | Değer |
|---|---|
| mAP50 | 0.801 |
| mAP50-95 | 0.462 |
| Precision | 0.792 |
| Recall | 0.774 |
| Inference | 3.5 ms/görüntü |

### Tracking Performansı (3 test sequence)
| Metrik | Değer | Açıklama |
|---|---|---|
| MOTA | 0.345 | Düşük — az veri, düşük çözünürlük |
| IDF1 | 0.083 | Düşük — ID karışması fazla |
| num_switches | 45 | Yanlış ID ataması |
| FPS | 35.1 | Gerçek zamanlı için yeterli |

> Düşük metrikler az veri (20/200 sequence) ve düşük çözünürlük (imgsz=320) nedeniyle. Pipeline mimarisi eksiksiz çalışmaktadır.

---

## İyileştirme Fırsatları

| İyileştirme | Beklenen Etki |
|---|---|
| 200 sequence ile eğitim | mAP 0.80 → 0.90+ |
| `imgsz=640` | Küçük drone tespiti iyileşir |
| SAHI + iyi model | FN azalır, MOTA artar |
| `max_dist` artır | ID_switch azalabilir |
| Daha fazla epoch | Model olgunlaşır |

---

## Teknik Detaylar

- **Framework:** Ultralytics YOLOv8s (pretrained ImageNet → fine-tuned Anti-UAV410)
- **Tracking:** Kalman Filtresi + Macar Algoritması
- **Dataset:** Anti-UAV410 (410 sequence, 640x512 IR termal)
- **Eğitim:** 20 epoch, imgsz=320, batch=16, Apple M2 MPS, ~5.9 saat
- **Inference:** 3.5ms/frame → 35+ FPS
- **Dil:** Python 3.11
