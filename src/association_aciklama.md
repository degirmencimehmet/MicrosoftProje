# Macar Algoritması — association.py Açıklaması

## Problem

YOLO her frame'de birden fazla tespit döndürebilir.
Elimizde birden fazla Kalman tracker'ı da olabilir.
Hangi tespit hangi tracker'a ait? Bunu bulmamız lazım.

```
Tracker'lar (Kalman tahminleri):
  T0 → (100, 200)
  T1 → (400, 300)

YOLO tespitleri:
  D0 → (102, 198)
  D1 → (405, 297)
  D2 → (250, 250)  ← muhtemelen false positive
```

---

## Adım 1: Maliyet Matrisi

Her tracker ile her tespit arasındaki piksel mesafesini hesapla.

```
         D0     D1     D2
T0  [  2.8   350    158  ]
T1  [  316     5.8  158  ]
```

Formül: `mesafe = sqrt((x1-x2)² + (y1-y2)²)`

```python
def build_cost_matrix(trackers, detections):
    cost = np.zeros((len(trackers), len(detections)))
    for i, t in enumerate(trackers):
        for j, d in enumerate(detections):
            cost[i, j] = np.sqrt((t[0]-d[0])**2 + (t[1]-d[1])**2)
    return cost
```

`np.zeros((2, 3))` → 2 tracker, 3 tespit için sıfır matris oluştur, sonra doldur.

---

## Adım 2: Macar Algoritması

`linear_sum_assignment(cost)` → toplam maliyeti minimum yapacak eşleştirmeyi bul.

```python
row_ind, col_ind = linear_sum_assignment(cost)
# row_ind = [0, 1]  → T0 ve T1
# col_ind = [0, 1]  → D0 ve D1 ile eşleşti
```

Yani: T0→D0, T1→D1

Brute force tüm kombinasyonları dener (2! = 2 seçenek):
- T0→D0, T1→D1: toplam maliyet = 2.8 + 5.8 = 8.6  ✓ en düşük
- T0→D1, T1→D0: toplam maliyet = 350 + 316 = 666

---

## Adım 3: max_dist Filtresi

Macar algoritması her tracker'ı bir tespitle eşleştirmek **zorunda**.
Ama en yakın tespit 300 piksel uzakta olsa bile eşleştirir — bu yanlış.

`max_dist=50` → 50 pikselden uzaksa eşleştirmeyi reddet:

```python
for r, c in zip(row_ind, col_ind):
    if cost[r, c] > max_dist:
        unmatched_detections.append(c)  # çok uzak, reddet
    else:
        matches.append((r, c))          # kabul et
```

---

## Adım 4: Eşleşmeyen Tespitler

D2 hiçbir tracker'la eşleşmedi (col_ind'de yok):

```python
for j in range(len(detections)):
    if j not in col_ind:
        unmatched_detections.append(j)
```

D2 → unmatched → false positive olarak at.

---

## Fonksiyonun Dönüş Değerleri

```python
matches, unmatched = associate(trackers, detections)

# matches          = [(0, 0), (1, 1)]  → (tracker_idx, detection_idx)
# unmatched        = [2]               → D2 eşleşmedi
```

---

## Özet Akış

```
Kalman tahminleri + YOLO tespitleri
        ↓
Maliyet matrisi oluştur (mesafe hesapla)
        ↓
Macar algoritması ile en iyi eşleştirmeyi bul
        ↓
max_dist filtresi uygula (çok uzakları reddet)
        ↓
matches → Kalman'ı update et
unmatched → false positive, at
```

---

## Kenar Durumlar

```python
if len(trackers) == 0 or len(detections) == 0:
    return [], list(range(len(detections)))
```

- Hiç tracker yoksa → tüm tespitler eşleşmedi
- Hiç tespit yoksa → hiçbir şey eşleşmedi
