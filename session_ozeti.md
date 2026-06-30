# Session Özeti

---

## 1. Notebook Tamamlandı
`goruntu_isleme_ders_notu.ipynb` tamamlandı. Özet: `ozet.md`

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

## 2. Proje Yapısı

```
Microsoft proje/
    dataset/                  ← Anti-UAV410 ham veri (git ignore)
        train/ (200 seq)
        val/   (90 seq)
        test/  (120 seq)
    yolo_dataset/             ← Tüm veri YOLO formatında (209K frame, git ignore)
    yolo_dataset_small/       ← Küçük subset (git ignore)
    runs/                     ← YOLO eğitim çıktıları (git ignore)
    src/
        kalman.py             ← Kalman filtresi
        association.py        ← Macar algoritması
        tracker.py            ← Durum makinesi
        visualize.py          ← Görselleştirme
        kalman_aciklama.md    ← (ileride)
        association_aciklama.md
        tracker_aciklama.md
        visualize_aciklama.md
    convert_to_yolo.py
    create_small_dataset.py
    train.py
    test_model.ipynb
    colab_train.ipynb
    .gitignore
    session_ozeti.md
    ozet.md
```

**Önemli:** Her zaman `python3.11` kullan. `python3` farklı bir versiyona işaret ediyor.

---

## 3. Anti-UAV410 Dataset

- Her sequence: sıralı `.jpg` frame'ler + `IR_label.json`
- JSON: `{"exist": [1,1,0,...], "gt_rect": [[x,y,w,h],...]}`
- `exist=1` → drone görünür, `gt_rect` → sol-üst köşe + genişlik/yükseklik
- Görüntü boyutu: **640x512** (IR termal)
- Train: 200 seq, 209011 frame | Val: 90 seq, 92623 | Test: 120 seq, 127069

---

## 4. YOLO Format Dönüşümü

`convert_to_yolo.py`:
- `exist=1` olan frame'leri seçer
- `[x,y,w,h]` → YOLO normalize: `x_c=(x+w/2)/640`, `y_c=(y+h/2)/512`
- Sembolik link oluşturur (disk tasarrufu)
- `data.yaml` oluşturur

Küçük subset (`create_small_dataset.py`):
- Train: 20 seq → 22199 frame
- Val: 10 seq → 10453 frame
- Test: 10 seq → 11620 frame

---

## 5. YOLOv8 Eğitimi

```python
model = YOLO("yolov8s.pt")
model.train(
    data="yolo_dataset_small/data.yaml",
    epochs=20, imgsz=320, batch=16,
    device="mps", patience=10, save_period=5
)
```

Karşılaşılan sorunlar:
| Sorun | Çözüm |
|---|---|
| `python3` torch bulamıyor | `python3.11` kullan |
| 209K frame → 7 saat/epoch | 20 sequence'lık küçük dataset |
| `batch=8` yavaş | `batch=16` |
| `imgsz=640` → 7 saat/epoch | `imgsz=320` |
| Drive yükleme kesildi | Mac'te local eğitim |

Tüm epoch sonuçları:
| Epoch | mAP50 | P | R |
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
| **17** | **0.801** | 0.792 | 0.774 |
| 18 | 0.780 | 0.966 | 0.687 |
| 19 | 0.787 | 0.962 | 0.706 |
| 20 | 0.789 | 0.964 | 0.702 |

**Final (best.pt = Epoch 17):** mAP50: 0.801, mAP50-95: 0.462, 3.5ms/görüntü
Model: `runs/uav_detector-5/weights/best.pt`

---

## 6. Model Testi (test_model.ipynb)

```python
model = YOLO("runs/uav_detector-5/weights/best.pt")
results = model(img_path, conf=0.3)

boxes = results[0].boxes
boxes.xyxy   # koordinatlar (x1,y1,x2,y2)
boxes.conf   # güven skoru
boxes.cls    # sınıf (0=uav)

# Görselleştir
annotated = results[0].plot()
plt.imshow(annotated[:,:,::-1])
```

`conf` → güven eşiği (düşük=çok tespit, yüksek=az ama emin)
`iou` → çakışan bbox temizleme eşiği
`w = x2-x1`, `h = y2-y1`

---

## 7. Kalman Filtresi (src/kalman.py)

**Amaç:** YOLO koordinatlarını yumuşatmak, drone kaybolunca tahminle devam etmek.

**State:** `[x, y, vx, vy]` — konum + hız

**Matrisler:**
| Matris | Boyut | Açıklama |
|---|---|---|
| F | 4x4 | Transition: `x_yeni = x + vx` |
| H | 2x4 | Measurement: state'ten sadece x,y seç |
| P | 4x4 | State kovaryansı, başta 1000 (belirsiz) |
| R | 2x2 | YOLO ölçüm gürültüsü = 10 |
| Q | 4x4 | Model gürültüsü = 0.1 |

**Kalman Gain (K):** R büyük → K küçük → tahminine güven. R küçük → K büyük → YOLO'ya güven.

**predict():**
```python
state = F @ state          # x_yeni = x + vx
P = F @ P @ F.T + Q        # belirsizlik artar
return state[:2]           # sadece x,y
```

**update(x, y):**
```python
z = [x, y]                          # YOLO ölçümü
S = H @ P @ H.T + R                 # toplam belirsizlik
K = P @ H.T @ inv(S)                # Kalman gain
state = state + K @ (z - H@state)   # ölçüm ile birleştir (innovation)
P = (I - K@H) @ P                   # belirsizlik azalır
```

**Döngü:** `predict → update → predict → update → ...`
YOLO kaybolursa: sadece `predict()` ile devam.

**Test sonucu:**
```
kalmanFilter(100, 200)
predict()        → [100, 200]
update(105, 203) → [104.97, 202.98]   # YOLO'ya yaklaştı
predict()        → [107.46, 204.47]   # vx≈5, vy≈3 öğrenildi

# 3 frame YOLO kaybolursa:
Frame 1: [119.95, 211.97]
Frame 2: [124.92, 214.95]
Frame 3: [129.90, 217.94]   # ~5 px/frame devam eder
```

---

## 8. Macar Algoritması (src/association.py)

**Problem:** N tracker, M tespit → hangisi hangisiyle eşleşmeli?

**Çözüm:**
1. Maliyet matrisi oluştur (her tracker-tespit arası mesafe)
2. `linear_sum_assignment` ile toplam maliyeti minimize eden eşleştirmeyi bul
3. `max_dist=50` filtresi: 50 pikselden uzak eşleştirmeleri reddet

```python
def build_cost_matrix(trackers, detections):
    cost = np.zeros((len(trackers), len(detections)))
    for i, t in enumerate(trackers):
        for j, d in enumerate(detections):
            cost[i,j] = sqrt((t[0]-d[0])² + (t[1]-d[1])²)
    return cost

def associate(trackers, detections, max_dist=50):
    cost = build_cost_matrix(trackers, detections)
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, unmatched = [], []
    for r, c in zip(row_ind, col_ind):
        if cost[r,c] > max_dist: unmatched.append(c)
        else: matches.append((r, c))
    # eşleşmeyen tespitler de unmatched'e
    return matches, unmatched
```

**Test:** T0→D0 (2.8px), T1→D1 (5.8px), D2 eşleşmedi → false positive

---

## 9. Durum Makinesi (src/tracker.py)

**State enum:**
```
ARANIYOR=0, KİLİTLİ=1, TAHMİN=2, KAYIP=3
```

**Geçişler:**
```
ARANIYOR → drone bulununca → KİLİTLİ
KİLİTLİ → kaybolunca → TAHMİN
TAHMİN  → tekrar bulununca → KİLİTLİ
TAHMİN  → missing_count > max_missing → KAYIP → sil
```

**Track sınıfı:** Her drone için: `id`, `kalmanFilter`, `state`, `missing_count`

**Tracker.update(detections) akışı:**
1. Tüm track'leri `predict()` et
2. `associate()` ile eşleştir
3. Eşleşenleri `update()` et → missing_count=0, KİLİTLİ
4. Eşleşmeyenleri: missing_count++, TAHMİN; limit aşılırsa KAYIP
5. Eşleşmeyen tespitler → yeni Track oluştur
6. KAYIP track'leri sil

**Test:**
```
Frame 1-2: [(0,KILITLI), (1,KILITLI)]
Frame 3-5: [(0,TAHMIN),  (1,KILITLI)]   # T0 kayboldu
Frame 6:   [(1,KILITLI)]                # T0 silindi
```

---

## 10. Görselleştirme (src/visualize.py)

```python
import cv2

COLORS = {
    "KILITLI":  (0, 255, 0),    # Yeşil  (BGR!)
    "TAHMIN":   (0, 165, 255),  # Turuncu
    "ARANIYOR": (255, 0, 0),    # Mavi
    "KAYIP":    (0, 0, 255),    # Kırmızı
}

def draw_tracks(frame, tracks):
    for track in tracks:
        x, y, vx, vy = track.kalman.state   # Kalman'dan konum al
        x, y = int(x), int(y)               # float → int
        color = COLORS.get(track.state.name, (255,255,255))
        cv2.rectangle(frame, (x-10,y-10), (x+10,y+10), color, 2)  # bbox
        label = f"ID:{track.id} {track.state.name}"
        cv2.putText(frame, label, (x-10,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
```

**Koordinat sistemi:** Görüntüde y aşağı artar → `y-10` sol **üst**, `y+10` sağ **alt** köşe.
**BGR:** OpenCV BGR kullanır. `(0,255,0)` → Yeşil (G=255).
**Matplotlib'de göstermek için:** `result[:,:,::-1]` ile BGR→RGB çevir.

---

## 11. Tam Pipeline (YOLO + Tracker + Visualize)

```python
# 1. YOLO tespiti
results = model(img_path, conf=0.3)
boxes = results[0].boxes

# 2. Merkez koordinatlarına çevir
detections = []
for box in boxes.xyxy:
    x1, y1, x2, y2 = box.tolist()
    detections.append(((x1+x2)/2, (y1+y2)/2))

# 3. Tracker güncelle
tracks = tracker.update(detections)

# 4. Çiz
frame = img[:,:,::-1].copy()
result = draw_tracks(frame, tracks)
plt.imshow(result[:,:,::-1])
```

---

## 12. Değerlendirme (src/evaluate.py)

**Fonksiyon:** `evaluate(tracker, model, sequences, conf=0.3)`

**Akış:**
1. Her sequence için tracker sıfırla
2. Her frame'de: ground truth al → YOLO çalıştır → tracker güncelle
3. `mm.MOTAccumulator` ile gt vs tahmin mesafelerini biriktir
4. Tüm frame'ler bittikten sonra MOTA, IDF1, FPS hesapla

**Metrikler:**
- **MOTA** = `1 - (FP + FN + ID_switch) / toplam_gt` → 1'e yakın iyi
- **IDF1** → doğru ID eşleşme oranı → 1'e yakın iyi
- **num_switches** → kaç kez yanlış ID atandı → düşük olmalı
- **FPS** → saniyede kaç frame işlendi

**İlk sonuçlar (3 sequence, imgsz=320, 20 seq eğitim):**
| Metrik | Değer | Yorum |
|---|---|---|
| MOTA | 0.345 | Düşük — model zayıf, çok drone kaçırıyor |
| IDF1 | 0.083 | Çok düşük — ID_switch fazla |
| num_switches | 45 | ID karışması fazla |
| FPS | 35.1 | İyi — gerçek zamanlı için yeterli |

**Düşük metrik nedenleri:**
- `imgsz=320` küçük → küçük drone'lar kaçırılıyor
- Sadece 20 sequence ile eğitim → model genelleşemiyor
- `max_dist=50` çok kısıtlayıcı olabilir

```python
# Kullanım
sequences = [(seq_klasör, json_yolu), ...]
tracker = Tracker(max_missing=3)
summary = evaluate(tracker, model, sequences, conf=0.3)
```

---

## 13. Proje Durumu

**Tamamlanan:**
- [x] `src/kalman.py` — Kalman filtresi
- [x] `src/association.py` — Macar algoritması
- [x] `src/tracker.py` — Durum makinesi
- [x] `src/visualize.py` — Görselleştirme
- [x] `src/evaluate.py` — MOTA, IDF1, FPS

**Sonraki Adımlar (Hafta 3):**
- [ ] Model iyileştirme: daha fazla sequence, `imgsz=640`
- [ ] SAHI ekle — küçük/uzak drone tespitini iyileştir
- [ ] Demo videosu — gerçek sequence üzerinde tam pipeline
- [ ] Metrik karşılaştırma tablosu (baseline vs iyileştirilmiş)

---

## 14. Önemli Notlar

- `.gitignore`: `dataset/`, `yolo_dataset/`, `yolo_dataset_small/`, `runs/`, `*.pt`
- Model: `runs/uav_detector-5/weights/best.pt` (22.5MB)
- Sembolik linkler Finder'da görünmez ama YOLO okuyabilir
- Mac'te `.DS_Store` klasör listesini bozar → `.startswith(".")` ile filtrele
- Açıklama dosyaları: `src/association_aciklama.md`, `src/tracker_aciklama.md`, `src/visualize_aciklama.md`, `src/visualize_aciklama.md`
- `python3` değil her zaman `python3.11` kullan
