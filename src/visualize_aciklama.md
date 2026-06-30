# visualize.py — Satır Satır Açıklama

## Import

```python
import cv2
```
OpenCV kütüphanesi. Görüntü üzerine çizim için kullanılıyor.
`matplotlib` değil, çünkü:
- `cv2` direkt piksel üzerine çizer, çok hızlı
- Video frame'leri için idealdir
- `matplotlib` her seferinde yeni figür açar, video için yavaş

---

## COLORS Sözlüğü

```python
COLORS = {
    "KILITLI":  (0, 255, 0),    # Yeşil
    "TAHMIN":   (0, 165, 255),  # Turuncu
    "ARANIYOR": (255, 0, 0),    # Mavi
    "KAYIP":    (0, 0, 255),    # Kırmızı
}
```

Her durum için farklı renk. Format **(B, G, R)** — OpenCV BGR kullanır, RGB değil.
Örnek: `(0, 255, 0)` → B=0, G=255, R=0 → Yeşil

Neden farklı renkler?
- Ekranda bakınca hangi drone'un hangi durumda olduğunu anında görmek için
- KİLİTLİ=yeşil (iyi), TAHMİN=turuncu (dikkat), KAYIP=kırmızı (tehlike)

---

## draw_tracks Fonksiyonu

```python
def draw_tracks(frame, tracks):
```
- `frame` → OpenCV BGR görüntüsü (numpy array)
- `tracks` → Tracker'dan gelen aktif Track objelerinin listesi

---

```python
    for track in tracks:
```
Her aktif track için döngü. Her track bir drone'u temsil eder.

---

```python
        x, y, vx, vy = track.kalman.state
```
Kalman filtresinin state vektörünü aç:
- `x, y` → drone'un tahmini merkez koordinatı (float)
- `vx, vy` → hız (çizimde kullanmıyoruz ama state'te var)

---

```python
        x, y = int(x), int(y)
```
Kalman float değer üretir (örn: 104.97), cv2 integer ister.
`int()` ile yuvarlıyoruz.

---

```python
        color = COLORS.get(track.state.name, (255, 255, 255))
```
Track'in durumuna göre renk seç.
- `track.state.name` → "KILITLI", "TAHMIN" gibi string döndürür
- `.get(key, default)` → eğer sözlükte yoksa beyaz `(255,255,255)` kullan

---

```python
        cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), color, 2)
```
Görüntü üzerine dikdörtgen çiz.

Parametreler:
- `frame` → üzerine çizilecek görüntü
- `(x-10, y-10)` → sol üst köşe
- `(x+10, y+10)` → sağ alt köşe
- `color` → renk (BGR)
- `2` → çizgi kalınlığı (piksel)

**Neden x-10, y-10?**
Kalman merkez nokta veriyor. 10 piksel çıkarıp ekleyerek etrafına 20x20 piksel kutu çiziyoruz.
Görüntü koordinat sistemi: y **aşağı** artar, yani:
- `y-10` → yukarı git → sol **üst** köşe
- `y+10` → aşağı git → sağ **alt** köşe

---

```python
        label = f"ID:{track.id} {track.state.name}"
```
Yazdırılacak etiket oluştur. Örnek: `"ID:0 KILITLI"`, `"ID:1 TAHMIN"`

---

```python
        cv2.putText(frame, label, (x-10, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
```
Görüntü üzerine metin yaz.

Parametreler:
- `frame` → üzerine yazılacak görüntü
- `label` → yazılacak metin
- `(x-10, y-15)` → metnin sol alt köşesi (bbox'ın 5 piksel üstü)
- `cv2.FONT_HERSHEY_SIMPLEX` → font tipi
- `0.5` → font boyutu
- `color` → renk
- `1` → çizgi kalınlığı

---

```python
    return frame
```
Bbox ve etiketler çizilmiş görüntüyü döndür.

---

## Matplotlib'de Gösterme

`cv2` BGR, `matplotlib` RGB bekler. Gösterirken çevirmek lazım:

```python
plt.imshow(result[:, :, ::-1])
```
`[:, :, ::-1]` → tüm satır ve sütunları al, kanal sırasını ters çevir (BGR→RGB)

---

## Özet Akış

```
Track listesi + Frame gelir
      ↓
Her track için:
  Kalman'dan (x, y) al
  Duruma göre renk seç
  Bbox çiz (20x20 piksel kutu)
  ID ve durum etiketi yaz
      ↓
Çizilmiş frame döner
```
