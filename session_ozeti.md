# Session Özeti

## 1. Notebook Tamamlandı
Önceki sessionlarda yapılan `goruntu_isleme_ders_notu.ipynb` notebook'u tamamlandı.
Tüm konuların özeti `ozet.md` dosyasına kaydedildi.

Kapsanan konular:
- BGR/RGB renk uzayı ve dönüşümü
- Parlaklık/kontrast ayarı ve CLAHE
- Gürültü ekleme ve Gaussian/Median Blur
- Sobel filtresi ile kenar tespiti
- Morfolojik işlemler (Erosion, Dilation, Opening)
- Thresholding (eşikleme)
- Kontur bulma ve Bounding Box
- IoU hesaplama
- Tam pipeline demo

---

## 2. Proje Yapısı Kuruldu

**Klasör yapısı:**
```
Microsoft proje/
    dataset/              ← Anti-UAV410 ham veri (git ignore edildi)
        train/            (200 sequence)
        val/              (90 sequence)
        test/             (120 sequence)
    yolo_dataset/         ← Tüm veri YOLO formatında (209K frame)
    yolo_dataset_small/   ← Küçük subset (20 train, 10 val, 10 test sequence)
    runs/                 ← YOLO eğitim çıktıları (git ignore edildi)
    convert_to_yolo.py    ← Dataset dönüştürme scripti
    create_small_dataset.py ← Küçük subset oluşturma scripti
    train.py              ← YOLOv8 eğitim scripti
    colab_train.ipynb     ← Colab versiyonu (Drive + GPU)
    .gitignore
```

---

## 3. Anti-UAV410 Dataset

**Format:**
- Her sequence klasöründe sıralı `.jpg` frame'ler ve `IR_label.json` var
- JSON formatı:
```json
{
  "exist": [1, 1, 0, ...],
  "gt_rect": [[x, y, w, h], [x, y, w, h], null, ...]
}
```
- `exist=1` → o frame'de drone görünür
- `gt_rect` → sol-üst köşe x,y + genişlik,yükseklik (piksel)
- Görüntü boyutu: **640x512** (IR termal kamera)

**Dataset istatistikleri:**
- Train: 200 sequence, 209011 frame (exist=1)
- Val: 90 sequence, 92623 frame
- Test: 120 sequence, 127069 frame

---

## 4. YOLO Format Dönüşümü

**convert_to_yolo.py** scripti:
- `exist=1` olan frame'leri seçer
- `[x, y, w, h]` piksel → YOLO normalize formatına çevirir:
  ```
  x_center = (x + w/2) / 640
  y_center = (y + h/2) / 512
  w_norm   = w / 640
  h_norm   = h / 512
  ```
- Görüntüleri kopyalamaz, **sembolik link** oluşturur (disk tasarrufu)
- `data.yaml` oluşturur (nc=1, names=['uav'])

**Küçük dataset (create_small_dataset.py):**
- Train: 20 sequence → 22199 frame
- Val: 10 sequence → 10453 frame
- Test: 10 sequence → 11620 frame

---

## 5. Kurulu Paketler

```
python3.11  ← kullanılması gereken python (torch bu versiyona kurulu)
ultralytics 8.4.78
sahi 0.12.1
opencv-python 4.13.0.92
filterpy 1.4.5
scipy 1.17.1
motmetrics
torch 2.12.1 (MPS destekli — Apple M2)
```

> ⚠️ `python3` komutu farklı bir Python'a işaret ediyor. Her zaman `python3.11` kullan.

---

## 6. YOLOv8 Eğitimi

**train.py ayarları:**
```python
model = YOLO("yolov8s.pt")   # small, pretrained
model.train(
    data="yolo_dataset_small/data.yaml",
    epochs=20,
    imgsz=320,      # 640 çok yavaş, 320 seçildi
    batch=16,
    device="mps",   # Apple Silicon GPU
    patience=10,
    save_period=5,
)
```

**Performans:**
- Her epoch ~15-16 dakika (MPS + workers=0 kısıtı)
- 20 epoch toplam ~5.5 saat

**Karşılaşılan sorunlar ve çözümler:**
| Sorun | Çözüm |
|---|---|
| `python3` torch bulamıyor | `python3.11` kullan |
| 209K frame → epoch başına 7 saat | Küçük dataset (20 seq) oluşturuldu |
| `batch=8` → daha yavaş | `batch=16`'ya döndürüldü |
| `imgsz=640` → 7 saat/epoch | `imgsz=320` yapıldı |
| Drive'a yükleme bağlantı kesildi | Mac'te local eğitime geçildi |

**Eğitim sonuçları (ilk 4 epoch):**
| Epoch | mAP50 | Precision | Recall |
|---|---|---|---|
| 1 | 0.795 | 0.784 | 0.747 |
| 2 | 0.696 | 0.943 | 0.630 |
| 3 | 0.765 | 0.933 | 0.684 |
| 4 | 0.787 | 0.967 | 0.685 |

**Tüm epoch sonuçları:**
| Epoch | mAP50 | Precision | Recall |
|---|---|---|---|
| 1 | 0.795 | 0.784 | 0.747 |
| 2 | 0.696 | 0.943 | 0.630 |
| 3 | 0.765 | 0.933 | 0.684 |
| 4 | 0.787 | 0.967 | 0.685 |
| 5 | 0.673 | 0.786 | 0.727 |
| 6 | 0.695 | 0.797 | 0.749 |
| 7 | 0.802 | 0.822 | 0.787 |
| 8 | 0.763 | 0.778 | 0.767 |
| 9 | 0.759 | 0.789 | 0.736 |
| 10 | 0.812 | 0.785 | 0.798 |
| 11 | 0.802 | 0.957 | 0.732 |
| 12 | 0.781 | 0.947 | 0.672 |
| 13 | 0.788 | 0.798 | 0.773 |
| 14 | 0.784 | 0.908 | 0.727 |
| 15 | 0.764 | 0.909 | 0.710 |
| 16 | 0.767 | 0.946 | 0.686 |
| 17 | **0.801** | 0.792 | 0.774 |
| 18 | 0.780 | 0.966 | 0.687 |
| 19 | 0.787 | 0.962 | 0.706 |
| 20 | 0.789 | 0.964 | 0.702 |

**Final sonuç (best.pt — Epoch 17):**
- mAP50: 0.801, mAP50-95: 0.462, Precision: 0.792, Recall: 0.774
- Inference hızı: 3.5ms/görüntü
- Model: `runs/uav_detector-5/weights/best.pt`
- Toplam süre: 5.893 saat

---

## 7. Model Testi (test_model.ipynb)

```python
from ultralytics import YOLO
model = YOLO("runs/uav_detector-5/weights/best.pt")
results = model(img_path, conf=0.5)
```

- `results[0].boxes.xyxy` → koordinatlar (x1,y1,x2,y2)
- `results[0].boxes.conf` → güven skoru
- `results[0].plot()` → bbox çizilmiş görüntü
- `w = x2-x1`, `h = y2-y1` ile genişlik/yükseklik hesaplanır
- `conf` parametresi ile güven eşiği ayarlanır, `iou` ile çakışan bbox'lar temizlenir

---

## 8. Kalman Filtresi (src/kalman.py)

**Amaç:** YOLO'nun gürültülü koordinatlarını yumuşatmak, drone kaybolunca tahminle devam etmek.

**State vektörü:** `[x, y, vx, vy]`

**Matrisler:**
- `F` (4x4) → transition matrix, `x_yeni = x + vx` mantığı
- `H` (2x4) → measurement matrix, state'ten sadece x,y seç
- `P` (4x4) → state kovaryansı, başlangıçta 1000 (belirsizlik yüksek)
- `R` (2x2) → YOLO ölçüm gürültüsü = 10
- `Q` (4x4) → model gürültüsü = 0.1 (tahminine daha çok güven)

**Kalman Gain (K):**
- R büyük → K küçük → tahminine güven
- R küçük → K büyük → YOLO'ya güven

**Döngü:**
```
predict() → update() → predict() → update() → ...
YOLO kaybolursa: sadece predict() ile devam et
```

**Test sonucu:**
```
tracker = kalmanFilter(100, 200)
predict()  → [100, 200]
update(105, 203) → [104.97, 202.98]   # YOLO'ya yaklaştı
predict()  → [107.46, 204.47]          # hız öğrenildi: vx≈5, vy≈3

# YOLO 3 frame kaybolursa:
Frame 1: [119.95, 211.97]
Frame 2: [124.92, 214.95]
Frame 3: [129.90, 217.94]   # ~5 piksel/frame ilerlemeye devam eder
```

---

## 9. Sonraki Adımlar (Hafta 2-3)

### Hafta 2 — Tracking
1. **Kalman Filtresi** (`src/kalman.py`)
   - State: `[x, y, vx, vy]`
   - Predict → Update döngüsü
   - YOLO gürültülü koordinatlarını yumuşatır, drone kaybedilince tahminle devam eder

2. **Macar Algoritması** (`src/association.py`)
   - Birden fazla tespit → hangi track'e ait eşleştirme
   - IoU + mesafe maliyet matrisi

3. **Durum Makinesi** (`src/tracker.py`)
   ```
   ARANIYOR → KİLİTLİ ↔ TAHMİN → KAYIP → ARANIYOR
   ```

4. **Görselleştirme** (`src/visualize.py`)
   - Bbox, track ID, durum etiketi

5. **Değerlendirme** (`src/evaluate.py`)
   - MOTA, IDF1, FPS

### Hafta 3 — Deney + Rapor
- SAHI ile küçük/uzak drone tespiti iyileştirme
- Metrik karşılaştırma tablosu
- Demo videosu

---

## 8. Önemli Notlar

- `.gitignore`'a eklenenler: `dataset/`, `yolo_dataset/`, `runs/`, `*.pt`
- Eğitim sonuçları: `runs/uav_detector-5/weights/best.pt`
- Sembolik linkler Finder'da önizleme göstermez ama YOLO okuyabilir
- Kalman filtresi tamamlandı, test edildi
- Sıradaki adım: Macar Algoritması (src/association.py)
- Birden fazla drone/tespit varsa eşleştirme problemi çözülecek
