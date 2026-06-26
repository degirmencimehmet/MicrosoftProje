# Görüntü İşleme Pipeline Özeti

Notebook'ta öğrenilen konulara göre, bir görüntü işleme sisteminde sırasıyla yapılması gereken adımlar:

---

## 1. Görüntü Okuma ve Renk Uzayı Dönüşümü

- Görüntü `cv2.imread()` ile BGR formatında okunur.
- Matplotlib ile göstermek için `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` ile RGB'ye çevrilir.
- Gri tona çevirmek için `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` kullanılır.

> **Not:** OpenCV'de kanal sırası BGR'dir. `[0, 0, 255]` yazdığında bu kırmızıdır (RGB'de `[255, 0, 0]`).

---

## 2. Ön İşleme (Preprocessing)

### Parlaklık / Kontrast Ayarı
- Piksel değerlerine sabit değer ekleyip/çıkararak parlaklık değiştirilir.
- `np.clip()` ile değerler 0–255 aralığında tutulur.

### Kontrast İyileştirme — CLAHE
- `cv2.createCLAHE()` ile lokal kontrast artırılır.
- Düşük kontrastlı görüntülerde nesneleri ön plana çıkarmak için kullanılır.
- Global histogram eşitlemeye göre üstündür çünkü lokal bölgelere ayrı ayrı uygular.

### Gürültü Ekleme (Augmentation)
- Piksel değerlerine rastgele sayılar eklenerek gürültü simüle edilir.
- Amacı: Sistemi sensör hataları, düşük ışık, termal dalgalanmalar gibi gerçek dünya koşullarına hazırlamak.
- `np.random.seed()` ile sonuçlar tekrar üretilebilir hale getirilir.

---

## 3. Gürültü Azaltma (Filtering)

| Yöntem | Açıklama | Ne Zaman Kullanılır |
|---|---|---|
| **Gaussian Blur** | Komşu piksellerin ağırlıklı ortalaması | Genel yumuşatma, kenar öncesi |
| **Median Blur** | Komşu piksellerin medyanı | Tuz-biber gürültüsünde daha etkili |

> Blur kernel boyutu arttıkça sınırlar belirsizleşir; sıcak bölgeler birbirine karışmaya başlar.

---

## 4. Kenar Tespiti

### Sobel Filtresi
- X ve Y yönündeki ani piksel değişimleri (gradyanlar) hesaplanır.
- `cv2.Sobel()` ile uygulanır.
- Kenar kuvveti: `magnitude = sqrt(Gx² + Gy²)`
- Termal görüntülerde sıcak/soğuk bölge sınırlarını bulmak için kullanılır.

---

## 5. Morfolojik İşlemler

Binary (siyah-beyaz) görüntü üzerinde uygulanır.

| İşlem | Açıklama | Kullanım |
|---|---|---|
| **Erosion** | Beyaz alanları küçültür | Gürültü noktalarını siler |
| **Dilation** | Beyaz alanları büyütür | Tespit edilen nesneleri genişletir |
| **Opening** | Erosion → Dilation | Küçük gürültü noktalarını temizler, nesneyi korur |
| **Closing** | Dilation → Erosion | Nesne içindeki delikleri kapatır |

---

## 6. Eşikleme (Thresholding)

- Piksel değeri belirlenen eşiğin üstündeyse beyaz (255), altındaysa siyah (0) yapılır.
- `cv2.threshold()` veya `cv2.adaptiveThreshold()` kullanılır.

> **Önemli:** Sabit eşik, uzaktaki zayıf sinyalli drone'u silebilir. Adaptif eşikleme veya CLAHE + threshold kombinasyonu daha güvenilirdir.

---

## 7. Kontur Bulma ve Bounding Box

- `cv2.findContours()` ile binary görüntüdeki beyaz bölgelerin sınırları bulunur.
- `cv2.boundingRect(contour)` ile her kontura bounding box çizilir.
- Küçük konturlar `cv2.contourArea()` ile filtrelenip gürültü elenebilir.

---

## 8. Değerlendirme — IoU (Intersection over Union)

- İki bounding box'ın ne kadar örtüştüğünü ölçer.
- Formül: `IoU = Kesişim Alanı / Birleşim Alanı`
- 0 → hiç örtüşme yok, 1 → tam örtüşme
- Tespitin doğruluğunu ölçmek için kullanılır.

---

## Tam Pipeline Akışı (Özet)

```
Ham Termal Frame
      ↓
Gri Tona Çevir
      ↓
Gürültü Azalt  (Gaussian / Median Blur)
      ↓
Kontrast İyileştir  (CLAHE)
      ↓
Eşikle  (Thresholding → Binary Görüntü)
      ↓
Morfolojik Temizlik  (Opening → gürültü sil)
      ↓
Kontur Bul  (findContours)
      ↓
Bounding Box Çiz
      ↓
IoU ile Değerlendir
```