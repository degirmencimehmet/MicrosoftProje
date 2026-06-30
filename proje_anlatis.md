# Termal İHA Takip Sistemi — Sıfırdan Anlatım

Bu belge, projenin başından sonuna kadar ne yaptığımızı, neden yaptığımızı ve nasıl çalıştığını sıfırdan açıklar.

---

## 1. Problem Nedir?

Termal kamera ile çekilen videolarda küçük bir insansız hava aracı (İHA/drone) var. Biz bu drone'u:
1. **Tespit etmek** istiyoruz — "o an görüntüde drone nerede?"
2. **Takip etmek** istiyoruz — "drone nereden nereye gidiyor, kaybolursa nerede?"
3. **Kilit almak** istiyoruz — drone bulununca üstünde sabit kal, kaybedersen tahminle devam et

Termal görüntü, normal RGB kameranın aksine ısı yaydıkça parlayan görüntüler üretir. Drone motor ısısı nedeniyle parlak görünür — bu avantaj. Ama görüntü gürültülü, kontrast düşük olabilir — bu dezavantaj.

---

## 2. Genel Pipeline

```
Termal Video Frame
      ↓
YOLO ile Tespit        → "Bu frame'de drone burada"
      ↓
Macar Algoritması      → "Bu tespit hangi track'e ait?"
      ↓
Kalman Filtresi        → "Koordinatları yumuşat, kaybolursa tahmin et"
      ↓
Durum Makinesi         → "ARANIYOR / KİLİTLİ / TAHMİN / KAYIP"
      ↓
Görselleştirme         → "Bbox ve durum ekrana çiz"
      ↓
Değerlendirme          → "MOTA, IDF1, FPS hesapla"
```

---

## 3. Veri Seti — Anti-UAV410

### Nedir?
410 adet video sequence'ından oluşan termal İHA veri seti.
- **Train:** 200 sequence (modeli öğretmek için)
- **Val:** 90 sequence (eğitim sırasında kontrol için)
- **Test:** 120 sequence (final değerlendirme için)

### Klasör Yapısı
```
dataset/
  train/
    01_1667_0001-1500/
      000001.jpg
      000002.jpg
      ...
      IR_label.json
    01_1751_0250-1750/
      ...
  val/
    ...
  test/
    ...
```

Her sequence bir video demek. İçinde sıralı jpg frame'ler ve bir adet `IR_label.json` var.

### JSON Format
```json
{
  "exist": [1, 1, 0, 1, ...],
  "gt_rect": [[306, 270, 14, 11], [313, 269, 14, 12], null, ...]
}
```

- `exist[i] = 1` → i. frame'de drone görünür
- `exist[i] = 0` → drone o frame'de yok (bulut arkası, çok uzak vb.)
- `gt_rect[i] = [x, y, w, h]` → drone'un o frame'deki gerçek konumu
  - `x, y` → bounding box'ın sol üst köşesi (piksel)
  - `w, h` → genişlik ve yükseklik (piksel)
  - `null` → exist=0 olduğunda konum yok

### Görüntü Boyutu
640x512 piksel — termal IR kamera çözünürlüğü.

---

## 4. YOLO Formatına Dönüştürme

### Neden Dönüştürme Gerekli?
YOLO kendi etiket formatını bekler. Bizim JSON'daki format farklı.

**JSON formatı:** `[x_sol_üst, y_sol_üst, genişlik, yükseklik]` → piksel değerleri

**YOLO formatı:** `class x_center y_center width height` → 0-1 arası normalize değerler

### Dönüşüm Formülü
```python
x_center = (x + w/2) / görüntü_genişliği   # 640
y_center = (y + h/2) / görüntü_yüksekliği  # 512
w_norm   = w / 640
h_norm   = h / 512
```

Neden normalize ediyoruz? YOLO farklı boyutlardaki görüntülere uygulanabilmesi için 0-1 arası değer bekler.

**Örnek:**
```
JSON:  [306, 270, 14, 11]
       x=306, y=270, w=14, h=11

x_center = (306 + 14/2) / 640 = 313/640 = 0.4891
y_center = (270 + 11/2) / 512 = 275.5/512 = 0.5381
w_norm   = 14/640 = 0.0219
h_norm   = 11/512 = 0.0215

YOLO label: 0 0.4891 0.5381 0.0219 0.0215
            ↑ class_id (0=uav)
```

### Sonuç Yapısı
```
yolo_dataset/
  train/
    images/ → görüntüler (sembolik link)
    labels/ → .txt etiket dosyaları
  val/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml → YOLO config dosyası
```

**Sembolik link nedir?** Dosyayı kopyalamak yerine "bu dosya şurada" diye işaret koymak. Disk tasarrufu sağlar.

### data.yaml
```yaml
path: /yolo_dataset
train: train/images
val: val/images
test: test/images
nc: 1              # sınıf sayısı
names: ['uav']     # sınıf adları
```

---

## 5. YOLOv8 ile Tespit

### YOLO Nedir?
"You Only Look Once" — görüntüyü bir kez geçirerek tüm nesneleri tespit eden derin öğrenme modeli. Gerçek zamanlı çalışabilecek kadar hızlı.

### Neden YOLOv8s?
- `s` = small → hafif model, Mac'te çalışabilir
- Pretrained → ImageNet üzerinde 80 sınıf öğrenmiş, biz üzerine drone öğreteceğiz

### Fine-Tuning Nedir?
Hazır eğitilmiş modeli kendi veri setimizle yeniden eğitmek. Model sıfırdan başlamak yerine genel görsel özellikleri biliyor, sadece "drone nasıl görünür" kısmını öğreniyor. Bu sayede:
- Daha az veri gerekir
- Daha hızlı öğrenir
- Daha iyi sonuç verir

### Eğitim Parametreleri
```python
model = YOLO("yolov8s.pt")   # pretrained model
model.train(
    data="data.yaml",
    epochs=20,     # kaç tur eğitim
    imgsz=320,     # görüntü boyutu (640 Mac'te çok yavaş)
    batch=16,      # her adımda kaç görüntü
    device="mps",  # Apple Silicon GPU
)
```

**epoch:** Tüm eğitim verisini bir kez görme turu. 20 epoch = 22199 görüntüyü 20 kez gördü.

**batch:** Her gradyan güncellemesinde kullanılan görüntü sayısı. 16 görüntü → hesapla → güncelle → 16 görüntü → ...

**imgsz=320:** Görüntüler 320x320'ye resize ediliyor. 640 kullanmak 4x daha yavaş ama daha iyi sonuç verir.

### Train / Val / Test Farkı
- **Train:** Model bu görüntüleri görür, hatalarından öğrenir, ağırlıklarını günceller
- **Val:** Model tahmin yapar ama ağırlık güncellemez — "ne kadar öğrendim?" sorusunu cevaplar. Her epoch sonunda mAP hesaplanır. En yüksek val mAP'ına sahip model `best.pt` olarak kaydedilir
- **Test:** Eğitim tamamen bittikten sonra, hiç görmediği verilerle gerçek performans ölçülür

### Metrikler
**mAP50 (mean Average Precision @ IoU 0.5):**
Modelin tespit doğruluğu. Tahmin edilen bbox ile gerçek bbox'ın %50'den fazla örtüşmesi gerekir.
- 0 → hiç doğru tespit yok
- 1 → tüm tespitler doğru

**Precision:** Tespit ettiklerinin kaçı gerçekten drone?
```
Precision = Doğru Tespit / Toplam Tespit
```
Yüksekse: "tespit ettim demişse gerçekten vardır"

**Recall:** Gerçek drone'ların kaçını buldu?
```
Recall = Doğru Tespit / Toplam Gerçek Drone
```
Yüksekse: "hiç kaçırmıyor"

**Precision-Recall dengesi:** Birini artırınca diğeri azalır. `conf` eşiği ile ayarlanır.

### Eğitim Sonuçları (best.pt = Epoch 17)
```
mAP50     = 0.801   → %80 doğruluk
mAP50-95  = 0.462
Precision = 0.792
Recall    = 0.774
Inference = 3.5ms/görüntü
```

---

## 6. Kalman Filtresi (src/kalman.py)

### Neden Gerekli?
YOLO her frame'de drone'un koordinatını veriyor ama bu koordinatlar gürültülü:
```
Gerçek konum: (100, 200)
YOLO frame 1: (102, 198)   ← 2 piksel sapma
YOLO frame 2: (97, 203)    ← 3 piksel sapma
YOLO frame 3: (göremedi)   ← kayıp!
```

Kalman iki problemi çözüyor:
1. Gürültülü koordinatları yumuşatır
2. Drone kaybolunca tahminle devam eder

### State Vektörü
Drone'un durumunu 4 değişkenle takip ediyoruz:
```
state = [x, y, vx, vy]
         konum   hız
```
- `x, y` → drone'un merkez koordinatı
- `vx, vy` → x ve y yönündeki hız (piksel/frame)

### Kalman'ın İki Adımı

**1. Predict (Tahmin):**
Önceki konuma ve hıza bakarak "drone şimdi nerede?" sorusunu cevaplar.
```
x_yeni  = x + vx
y_yeni  = y + vy
vx_yeni = vx      (hız değişmez varsayımı)
vy_yeni = vy
```

Matris formunda (F = transition matrix):
```
[1, 0, 1, 0]   [x ]   [x + vx]
[0, 1, 0, 1] × [y ] = [y + vy]
[0, 0, 1, 0]   [vx]   [  vx  ]
[0, 0, 0, 1]   [vy]   [  vy  ]
```

**2. Update (Güncelleme):**
YOLO'dan yeni ölçüm gelince, tahmin ile ölçümü akıllıca birleştir.
```
innovation = ölçüm - tahmin       (fark ne kadar?)
state_yeni = tahmin + K × innovation
```
`K` = Kalman Gain → YOLO'ya mı güven, tahminine mi?

### Matrisler ve Anlamları
| Matris | Boyut | Açıklama |
|---|---|---|
| F | 4x4 | Transition: bir sonraki state'i hesaplar |
| H | 2x4 | Measurement: state'ten [x,y] seçer, vx/vy'yi görmezden gelir |
| P | 4x4 | State kovaryansı: ne kadar belirsiziz? (başta 1000, zamanla azalır) |
| R | 2x2 | YOLO'nun gürültüsü (=10, büyük → YOLO'ya az güven) |
| Q | 4x4 | Fizik modelimizin gürültüsü (=0.1, küçük → tahminine güven) |

**H matrisi neden 2x4?**
YOLO sadece `[x, y]` veriyor (2 değer), state `[x, y, vx, vy]` (4 değer).
H state'ten sadece x ve y'yi seçiyor:
```
[1, 0, 0, 0]   → x'i al
[0, 1, 0, 0]   → y'yi al
```

**Kalman Gain mantığı:**
- R büyük (YOLO güvenilmez) → K küçük → tahminine yaslan
- R küçük (YOLO güvenilir) → K büyük → YOLO'ya güven

### Test Sonucu
```python
tracker = kalmanFilter(100, 200)
# YOLO 3 frame drone'u göremezse:
predict() → [119.95, 211.97]   # vx≈5, vy≈3 öğrenilmişti
predict() → [124.92, 214.95]   # ~5px/frame devam ediyor
predict() → [129.90, 217.94]
```
YOLO göremese bile Kalman drone'un nerede olduğunu tahmin ediyor.

---

## 7. Macar Algoritması (src/association.py)

### Problem
Birden fazla drone ve birden fazla tespit olabilir:
```
Tracker'lar:    T0(100,200)  T1(400,300)
YOLO tespitler: D0(102,198)  D1(405,297)  D2(250,250)
```
Hangi tespit hangi tracker'a ait?

### Çözüm: Maliyet Matrisi + Macar Algoritması
**1. Maliyet matrisi:** Her tracker ile her tespit arasındaki piksel mesafesi
```
         D0    D1    D2
T0  [  2.8  350   158 ]
T1  [  316    5.8  158 ]
```

**2. Macar algoritması:** Toplam maliyeti minimum yapan eşleştirmeyi bul
```
T0→D0, T1→D1: maliyet = 2.8 + 5.8 = 8.6   ← en düşük ✓
T0→D1, T1→D0: maliyet = 350 + 316 = 666
```

**3. max_dist filtresi:** 50 pikselden uzak eşleştirmeleri reddet
D2 eşleşmedi → false positive → at

### Sonuç
```python
matches          = [(0,0), (1,1)]   # T0→D0, T1→D1
unmatched_dets   = [2]              # D2 kimseyle eşleşmedi
```

---

## 8. Durum Makinesi (src/tracker.py)

### Durumlar
```
ARANIYOR = 0   → drone henüz bulunamadı
KİLİTLİ = 1   → drone tespit edildi, takip ediliyor
TAHMİN  = 2   → YOLO göremedi, Kalman tahminle devam ediyor
KAYIP   = 3   → çok uzun süre kayıp, track silindi
```

### Geçişler
```
ARANIYOR ──tespit gelince──→ KİLİTLİ
KİLİTLİ ──tespit kaybolunca──→ TAHMİN
TAHMİN  ──tespit tekrar gelince──→ KİLİTLİ
TAHMİN  ──missing_count > 3──→ KAYIP ──→ (track silindi, yeni tespit gelince ARANIYOR)
```

### missing_count
Her frame'de drone görülemezse sayaç artar:
```
Frame 10: YOLO göremedi → missing_count=1 → TAHMİN
Frame 11: YOLO göremedi → missing_count=2 → TAHMİN
Frame 12: YOLO göremedi → missing_count=3 → TAHMİN
Frame 13: YOLO göremedi → missing_count=4 → KAYIP → sil
```
YOLO tekrar görürse: `missing_count=0`, KİLİTLİ'ye dön.

### Track Sınıfı
Her drone için bir Track objesi:
```python
class Track:
    id           # benzersiz numara (0, 1, 2, ...)
    kalman       # bu drone'un Kalman filtresi
    state        # şu anki durum (KİLİTLİ, TAHMİN vb.)
    missing_count # kaç frame'dir kayıp
```

### Tracker.update() Her Frame'de Şunu Yapar
1. Tüm track'leri `predict()` et → "bu frame'de neredeler?"
2. `associate()` ile YOLO tespitlerini eşleştir
3. Eşleşen track'leri `update()` et → Kalman güncelle, KİLİTLİ yap
4. Eşleşmeyen track'ler: missing_count++, TAHMİN; limit aşılırsa KAYIP
5. Eşleşmeyen tespitler → yeni drone, yeni Track oluştur
6. KAYIP track'leri listeden sil

### Test Sonucu
```
Frame 1-2: T0=KİLİTLİ, T1=KİLİTLİ   (2 drone görünüyor)
Frame 3-5: T0=TAHMİN,  T1=KİLİTLİ   (T0 kayboldu, Kalman devam ediyor)
Frame 6:   T1=KİLİTLİ                (T0 KAYIP→silindi)
```

---

## 9. Görselleştirme (src/visualize.py)

Her track için görüntü üzerine çizim:
- **Bounding box:** Drone'un etrafına renkli kutu
- **Etiket:** "ID:0 KİLİTLİ" gibi metin

```python
COLORS = {
    "KİLİTLİ":  (0, 255, 0),    # Yeşil  → iyi
    "TAHMİN":   (0, 165, 255),  # Turuncu → dikkat
    "ARANIYOR": (255, 0, 0),    # Mavi
    "KAYIP":    (0, 0, 255),    # Kırmızı → tehlike
}
```

**Önemli:** OpenCV BGR formatı kullanır, RGB değil.
- Normal: Kırmızı = `(255, 0, 0)` RGB
- OpenCV: Kırmızı = `(0, 0, 255)` BGR

**Koordinat sistemi:** Görüntüde y aşağı artar (matematikten farklı):
- `y-10` → yukarı git → sol **üst** köşe
- `y+10` → aşağı git → sağ **alt** köşe

---

## 10. SAHI (src/detector.py)

### Problem
Drone görüntüde çok küçük (14x11 piksel). YOLO 320x320'ye resize ederken daha da küçülüyor → kaçırılıyor.

### Çözüm: Slice and Infer
Görüntüyü küçük parçalara böl, her parçada YOLO çalıştır, birleştir.
```
640x512 görüntü → 256x256'lık slice'lara böl
Her slice 320x320'ye resize → drone görece büyür
Her slice'ta YOLO çalıştır
Sonuçları birleştir (NMS ile çakışanları temizle)
```

### overlap=0.2
Slice'lar %20 üst üste biner. Neden? Drone tam sınırda olursa ikiye bölünmeden görülsün.

### Sonuç
Normal YOLO: 1 tespit
SAHI: 4 tespit (ama model zayıf olduğundan false positive fazla)

---

## 11. Değerlendirme (src/evaluate.py)

### MOTA (Multiple Object Tracking Accuracy)
```
MOTA = 1 - (FP + FN + ID_switch) / toplam_gerçek_nesne
```
- **FP (False Positive):** Drone olmayan yeri işaretleme
- **FN (False Negative):** Drone varken görememe
- **ID_switch:** Aynı drone'a farklı ID verme
- 1'e yakın → iyi, 0'a yakın → kötü

### IDF1
Doğru ID eşleşme oranı. 1'e yakın iyi.

### FPS
Saniyede kaç frame işlenebiliyor. Gerçek zamanlı için 25+ gerekir.

### Sonuçlar
```
MOTA         = 0.345   → düşük (model zayıf)
IDF1         = 0.083   → çok düşük (ID karışması)
num_switches = 45      → 45 kez yanlış ID
FPS          = 35.1    → iyi!
```

### Neden Düşük?
1. Sadece 20 sequence ile eğitim (200'den)
2. `imgsz=320` düşük çözünürlük
3. Küçük drone'lar kaçırılıyor → FN fazla

---

## 12. Projenin Eksikleri ve İyileştirmeler

| İyileştirme | Beklenen Etki |
|---|---|
| 200 sequence ile eğitim | mAP 0.80 → 0.90+ |
| imgsz=640 | Küçük drone tespiti iyileşir |
| SAHI + iyi model | FN azalır, MOTA artar |
| max_dist artır | ID_switch azalabilir |

---

## 13. Dosya Yapısı

```
Microsoft proje/
  src/
    kalman.py              ← Kalman filtresi
    association.py         ← Macar algoritması
    tracker.py             ← Durum makinesi
    visualize.py           ← Görselleştirme
    detector.py            ← SAHI + YOLO wrapper
    evaluate.py            ← MOTA, IDF1, FPS
    *_aciklama.md          ← Her dosyanın detaylı açıklaması
  dataset/                 ← Anti-UAV410 ham veri
  yolo_dataset_small/      ← YOLO formatı (20 seq)
  runs/uav_detector-5/
    weights/
      best.pt              ← En iyi model (Epoch 17)
      last.pt              ← Son epoch modeli
  train.py                 ← Eğitim scripti
  convert_to_yolo.py       ← Dataset dönüştürme
  create_small_dataset.py  ← Küçük subset oluşturma
  test_model.ipynb         ← Test ve görselleştirme
```
