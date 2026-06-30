# tracker.py — Satır Satır Açıklama

## Import'lar

```python
from enum import Enum
from src.kalman import kalmanFilter
from src.association import associate
```
- `Enum` → durum isimlerini güvenli integer'lara bağlamak için
- `kalmanFilter` → her drone'un konumunu tahmin etmek için
- `associate` → Kalman tahminleri ile YOLO tespitlerini eşleştirmek için

---

## State (Durum) Sınıfı

```python
class State(Enum):
    ARANIYOR = 0
    KILITLI  = 1
    TAHMIN   = 2
    KAYIP    = 3
```

Her drone'un içinde bulunabileceği 4 durum. Enum kullandık çünkü:
- `State.KILITLI` yazmak `"KILITLI"` yazmaktan güvenli — yazım hatası olursa Python hemen uyarır
- Aslında sayılar: ARANIYOR=0, KILITLI=1, TAHMIN=2, KAYIP=3

Durum geçişleri:
```
ARANIYOR → drone bulununca → KİLİTLİ
KİLİTLİ → drone kaybolunca → TAHMİN
TAHMİN  → drone tekrar bulununca → KİLİTLİ
TAHMİN  → çok uzun süre kayıpsa → KAYIP
KAYIP   → track silinir → ARANIYOR (yeni track açılır)
```

---

## Track Sınıfı

Her drone için bir `Track` objesi oluşturulur.

```python
class Track:
    def __init__(self, x, y, track_id):
```
Drone ilk görüldüğünde çağrılır. `x, y` → ilk konum, `track_id` → bu drone'un kimliği (0, 1, 2...)

```python
        self.id = track_id
```
Her drone'un benzersiz ID'si. Ekranda "ID:0", "ID:1" diye göstermek için kullanılır.

```python
        self.kalman = kalmanFilter(x, y)
```
Bu drone için ayrı bir Kalman filtresi başlat. State: `[x, y, vx=0, vy=0]`

```python
        self.state = State.KILITLI
```
Drone ilk görüldüğünde direkt KİLİTLİ'ye alıyoruz — zaten tespit edildi.

```python
        self.missing_count = 0
```
Kaç frame boyunca YOLO bu drone'u göremedi sayacı. 0'dan başlar.

---

### predict() metodu

```python
    def predict(self):
        return self.kalman.predict()
```
Kalman filtresinin predict adımını çalıştır — drone'un bir sonraki frame'deki tahmini konumunu döndür.
`F × state` matris çarpımı burada gerçekleşir: `x_yeni = x + vx`, `y_yeni = y + vy`

---

### update() metodu

```python
    def update(self, x, y):
```
YOLO'dan yeni bir tespit geldiğinde çağrılır.

```python
        self.kalman.update(x, y)
```
Kalman'ın update adımını çalıştır — YOLO ölçümünü Kalman tahminiyle birleştir, hızı güncelle.

```python
        self.missing_count = 0
```
Drone tekrar görüldü → sayacı sıfırla.

```python
        self.state = State.KILITLI
```
TAHMİN veya başka bir durumdan gelsek bile, tespit geldi → KİLİTLİ'ye dön.

---

## Tracker Sınıfı

Tüm Track'leri yöneten ana sınıf. Her frame'de bir kez çağrılır.

```python
class Tracker:
    def __init__(self, max_missing=3):
```

```python
        self.tracks = []
```
Aktif track'lerin listesi. Başta boş.

```python
        self.next_id = 0
```
Yeni track oluşturulduğunda verilecek ID. Her seferinde 1 artar: 0, 1, 2, 3...

```python
        self.max_missing = max_missing
```
Kaç frame üst üste kayıp olursa KAYIP'a geçilsin. Default=3.

---

### update() metodu — Ana Döngü

Her frame'de YOLO tespitlerini alır, track'leri günceller.

```python
    def update(self, detections):
```
`detections` → o frame'de YOLO'nun bulduğu `[(x1,y1), (x2,y2), ...]` listesi.

---

**Adım 1: Tahmin**
```python
        predictions = []
        for t in self.tracks:
            predictions.append(t.predict())
```
Her aktif track için Kalman'ın bu frame'deki tahminini al.
Sonuç: `[(100, 200), (400, 300)]` gibi bir liste.

---

**Adım 2: Eşleştirme**
```python
        matches, unmatched_detections = associate(predictions, detections)
```
Macar algoritması ile Kalman tahminleri ve YOLO tespitlerini eşleştir.
- `matches` → `[(0,0), (1,1)]` gibi (tracker_idx, detection_idx) çiftleri
- `unmatched_detections` → hiçbir track'e yakın olmayan tespit indexleri

---

**Adım 3: Eşleşen track'leri güncelle**
```python
        for t_index, d_index in matches:
            self.tracks[t_index].update(*detections[d_index])
```
Her eşleşen (tracker, tespit) çifti için Kalman'ı güncelle.
`*detections[d_index]` → `(102, 198)` tuple'ını açarak `update(102, 198)` olarak gönderir.

---

**Adım 4: Eşleşmeyen track'leri işle**
```python
        matched_track_ids = [t_idx for t_idx, _ in matches]
        for i, track in enumerate(self.tracks):
            if i not in matched_track_ids:
                track.missing_count += 1
                track.state = State.TAHMIN
                if track.missing_count > self.max_missing:
                    track.state = State.KAYIP
```
- Önce eşleşen track ID'lerini topla
- Eşleşmeyen her track için: sayacı artır, TAHMİN'e al
- Sayaç max_missing'i geçtiyse: KAYIP'a al

---

**Adım 5: Yeni drone**
```python
        for d_idx in unmatched_detections:
            x, y = detections[d_idx]
            self.tracks.append(Track(x, y, self.next_id))
            self.next_id += 1
```
Hiçbir track'e eşleşmeyen tespit → yeni drone görüldü → yeni Track oluştur, ID ver.

---

**Adım 6: KAYIP track'leri sil**
```python
        self.tracks = [t for t in self.tracks if t.state != State.KAYIP]
```
KAYIP durumundaki track'leri listeden çıkar. 
List comprehension: "KAYIP olmayanları tut" mantığı.

---

**Adım 7: Döndür**
```python
        return self.tracks
```
Güncel aktif track listesini döndür. Her track'in `.id`, `.state`, `.kalman.state` bilgilerine erişilebilir.

---

## Özet Akış (Her Frame)

```
YOLO tespitleri gelir
        ↓
Tüm track'lerin Kalman tahmini alınır
        ↓
Macar algoritması ile eşleştirme yapılır
        ↓
┌─── Eşleşen track ───────────────────────────────────┐
│  Kalman.update() → missing_count=0 → KİLİTLİ       │
└──────────────────────────────────────────────────────┘
┌─── Eşleşmeyen track ────────────────────────────────┐
│  missing_count++ → TAHMİN                           │
│  missing_count > max_missing → KAYIP → sil          │
└──────────────────────────────────────────────────────┘
┌─── Eşleşmeyen tespit ───────────────────────────────┐
│  Yeni Track oluştur → KİLİTLİ                       │
└──────────────────────────────────────────────────────┘
        ↓
Güncel track listesi döner
```
