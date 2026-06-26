# thermal-uav-lockon-tracker — Proje Checklist

> **Proje:** Termal Görüntülerde Küçük Hava Hedeflerinin Kalman Filtresi ve Yeniden Yakalama ile Kilitlenmeli Takibi  
> **Süre:** ~3 hafta | **Dil:** Python | **Repo adı:** `thermal-uav-lockon-tracker`

---

## HAZIRLIK

- [ ] **Repo yap** — GitHub'da `thermal-uav-lockon-tracker` adıyla yeni repo oluştur
- [ ] **Klasör yapısını kur** — aşağıdaki yapıyı oluştur:
  ```
  thermal-uav-lockon-tracker/
  ├── README.md
  ├── requirements.txt
  ├── .gitignore          # data/ klasörünü ekle
  ├── data/               # veri seti buraya (git'e gitmiyor)
  ├── src/
  │   ├── detector.py     # YOLO + SAHI sarmalayıcı
  │   ├── kalman.py       # Kalman filtresi (kendi implementasyonun)
  │   ├── association.py  # Hungarian / nearest-neighbor eşleştirme
  │   ├── tracker.py      # iz yönetimi + durum makinesi + lock-on
  │   ├── evaluate.py     # metrikler, baseline karşılaştırması
  │   └── visualize.py    # video üstüne çizim
  ├── experiments/        # deney scriptleri ve sonuç grafikleri
  ├── report/             # teknik rapor
  └── demo/               # çıktı demo videoları
  ```
- [ ] **Sanal ortam kur** — `python -m venv venv`
- [ ] **Kütüphaneleri yükle:**
  ```bash
  pip install ultralytics opencv-python numpy scipy filterpy sahi matplotlib motmetrics
  pip freeze > requirements.txt
  ```
- [ ] **Veri setini indir** — Anti-UAV dataset (termal drone videoları + etiketler)
- [ ] **Veriyi incele** — birkaç frame'i OpenCV ile aç, etiket formatını anla

---

## HAFTA 1 — Tespit Pipeline'ı

**Hedef:** Termal video üstünde kare kare bounding box çizebiliyor olmak.

### Adım 1.1 — Veriyi yükle ve göster
- [ ] OpenCV ile video/frame dizisini oku
- [ ] Ground truth (GT) kutularını frame üstüne çiz
- [ ] Gözle: hedef kaç piksel? Nerede? Kaç frame görünür?

### Adım 1.2 — Temel YOLO tespiti
- [ ] `yolov8n.pt` (veya `yolov8s.pt`) ile baseline çıkarım yap
- [ ] Termalde kaç tane hedef yakalanıyor gözlemle (muhtemelen çok az)
- [ ] `detector.py` iskeletini yaz: `detect(frame) -> List[bbox]`

### Adım 1.3 — Fine-tuning (gerekirse)
- [ ] Anti-UAV etiketlerini YOLO formatına çevir (`class x_center y_center w h`, normalize)
- [ ] `antiuav.yaml` dataset config dosyası yaz
- [ ] `imgsz=1280` ile 50 epoch fine-tune yap
- [ ] Val metriklerini (mAP@0.5) kaydet

### Adım 1.4 — SAHI slicing inference
- [ ] SAHI ile sliced prediction uygula (slice: 512x512, overlap: 0.2)
- [ ] **SAHI öncesi vs sonrası recall karşılaştırması** → ilk grafik
- [ ] `detector.py`'ı SAHI destekli hale getir

### Hafta 1 Çıktısı
- [ ] `detector.py` çalışıyor: `detect(frame)` → kutular döndürüyor
- [ ] Demo video: bir video üstünde kutu çiziliyor
- [ ] README'ye hafta 1 notlarını ekle

---

## HAFTA 2 — Takip, Durum Makinesi, Kilitlenme

**Hedef:** Hedefe kilitlenip, kaybolduğunda Kalman tahminiyle takibi sürdüren sistem.

### Adım 2.1 — Kalman filtresini yaz (kendi implementasyonun!)
- [ ] `kalman.py` — sıfırdan yaz, filterpy'dan BAKMA (önce anla, sonra kullan)
  - Durum: `[x, y, vx, vy]`
  - `predict()` → sabit hız modeli ile tahmin
  - `update(z)` → ölçümle güncelle (Kalman kazancı hesapla)
- [ ] Matematiği anla: F, H, Q, R matrislerinin ne anlama geldiğini söyleyebil
- [ ] Basit test: sabit hızla giden bir noktayı filtrele, gürültüyü azalttığını göster

### Adım 2.2 — Data association
- [ ] `association.py` — Hungarian algoritması
  - İz tahminleri ile yeni tespitler arasında maliyet matrisi kur (Öklid uzaklık)
  - `scipy.optimize.linear_sum_assignment` kullan
  - **Gating**: maliyet > eşik ise eşleşmeyi reddet
- [ ] Neden Hungarian? "Minimum toplam maliyet" sorusunu beyaz tahtada açıklayabilir ol

### Adım 2.3 — İz yönetimi + durum makinesi
- [ ] `tracker.py` — her iz için:
  - Bir KalmanTracker instance'ı
  - Durum: `ARANIYOR | KİLİTLİ | TAHMİN | KAYIP`
  - `miss_count` sayacı
- [ ] Her frame döngüsü:
  1. Tüm izler için `predict()`
  2. Tespitlerle eşleştir (association)
  3. Eşleşen iz: `update()`, `miss_count=0`, durum=`KİLİTLİ`
  4. Eşleşmeyen iz: `miss_count+=1`, durum=`TAHMİN`
     - `miss_count > N` ise durum=`KAYIP`, izi sil
  5. Eşleşmeyen tespit: yeni iz başlat
- [ ] N hiperparametresini dene: 5, 10, 15, 30

### Adım 2.4 — Lock-on policy
- [ ] Birden çok iz varsa tek birini seç (politika: en yüksek güven skoru)
- [ ] Seçilen izi "kilitli hedef" olarak sürekli raporla
- [ ] Kaybolursa yeniden yakalama mantığı: yeni tespit geldikten sonra lock-on'u yenile

### Hafta 2 Çıktısı
- [ ] `tracker.py` çalışıyor: kilitli=**yeşil kutu**, tahmin=**sarı kutu**
- [ ] Hedef 10+ frame kaybolunca bile iz devam ediyor
- [ ] Demo video kaydedildi

---

## HAFTA 3 — Deney, Ölçüm, Rapor

**Hedef:** Yöntemin işe yaradığını kanıtlamak ve yazmak.

### Adım 3.1 — Baseline kur
- [ ] `evaluate.py`'da "sadece tespit" modu: takip katmanı yok, her frame bağımsız
- [ ] Aynı video üstünde baseline vs tracker karşılaştırması

### Adım 3.2 — Metrikler
- [ ] **Tespit:** precision, recall (SAHI'li / SAHI'siz)
- [ ] **Takip:** `motmetrics` ile MOTA, IDF1, ID switch sayısı
- [ ] **Hız:** FPS ölçümü (gerçek zamanlı mı?)

### Adım 3.3 — Occlusion deneyi (projenin değerini kanıtlar)
- [ ] Belirli frame aralıklarında GT kutularını sil (hedefi "gizle")
- [ ] Baseline: kilidi anında kaybeder
- [ ] Tracker: N frame boyunca tahminle devam eder, geri gelince yakalar
- [ ] **"Tahmin toleransı N vs kaç frame kaybı tolere edilir" eğrisi** → ana grafik

### Adım 3.4 — Hiperparametre taraması
- [ ] N (miss toleransı): [5, 10, 15, 30] → MOTA grafiği
- [ ] Gating eşiği: [30px, 50px, 80px, 100px] → ID switch grafiği
- [ ] Güven eşiği: [0.2, 0.3, 0.4, 0.5] → precision/recall tradeoff

### Adım 3.5 — Teknik rapor (4-6 sayfa)
- [ ] Giriş: Problem ve motivasyon
- [ ] İlgili çalışmalar: SORT, DeepSORT, küçük nesne tespiti
- [ ] Yöntem: YOLO+SAHI, Kalman denklemleri, Hungarian, durum makinesi diyagramı, lock-on
- [ ] Deneyler: veri seti, metrikler, baseline açıklaması
- [ ] Sonuçlar: tablolar + 4 grafik (SAHI etkisi, occlusion, N taraması, FPS)
- [ ] Tartışma/Sınırlar: nerede başarısız oluyor
- [ ] Karmaşıklık analizi: Kalman O(1)/iz, Hungarian O(n³)

### Adım 3.6 — Repo ve demo
- [ ] README: proje amacı, kurulum, kullanım, örnek çıktı
- [ ] Demo video: kilidi gösteren, renkli kutu çizen, metrik overlay'li
- [ ] `.gitignore` kontrol: `data/` git'e gitmiyor
- [ ] GitHub reposu temizlendi, commit mesajları düzgün

---

## TEMEL KAVRAMLAR (mülakat hazırlığı)

Her birini kendi cümlerinle açıklayabilmeden ilerleme:

- [ ] Detection vs Tracking farkı nedir?
- [ ] Tracking-by-detection paradigması nasıl çalışır?
- [ ] Kalman: predict adımı ne yapar? update ne yapar? Neden ikisi var?
- [ ] F, H, Q, R matrislerinin fiziksel anlamı nedir?
- [ ] "Tespit 1 frame ıskalandığında ne olur?" sorusuna cevap
- [ ] Hungarian algoritması neden minimum maliyet atar?
- [ ] Gating neden gerekli?
- [ ] Durum makinesi: 4 durum ve geçiş koşulları
- [ ] Lock-on policy neden gerekli?
- [ ] SAHI slicing neden küçük hedefte işe yarıyor?
- [ ] MOTA ve IDF1 neyi ölçüyor?

---

## DURUM MAKİNESİ

```
          tespit geldi
ARANIYOR ──────────────► KİLİTLİ
    ▲                        │
    │     miss_count > N     │ tespit ıskalandı
    │                        ▼
  KAYIP ◄────────────── TAHMİN
              miss_count > N  │
                              │ tespit geldi
                              ▼
                          KİLİTLİ
```

---

*Son güncelleme: 2026-06-25*
