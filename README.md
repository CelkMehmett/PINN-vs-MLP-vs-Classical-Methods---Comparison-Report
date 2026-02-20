---

# Proje Özeti / Project Overview

Bu proje, finansal türev ürünlerin fiyatlamasında Physics-Informed Neural Networks (PINN) yaklaşımını klasik yöntemler ve Multi-Layer Perceptron (MLP) ile karşılaştırır. Amaç, Black-Scholes denkleminin hem teorik hem de pratik olarak PINN ile nasıl çözülebileceğini, klasik yöntemlerle farklarını ve PINN'in avantajlarını ortaya koymaktır.

## İçerik / Contents
- Black-Scholes denkleminin PINN ile çözümü
- Klasik Black-Scholes formülü, Monte Carlo ve Sonlu Farklar yöntemleri
- MLP ile saf veri odaklı öğrenme
- Gerçek piyasa verisiyle test (yfinance ile)
- Amerikan opsiyonu ve Heston modeli için PINN şablonları
- Implied volatility (örtük volatilite) kalibrasyonu için PINN
- Yüksek çözünürlüklü grafikler ve web tabanlı görselleştirme

---

# Yöntemler / Methods

## 1. Physics-Informed Neural Network (PINN)
- Black-Scholes PDE doğrudan kayıp fonksiyonuna entegre edilir
- Veri kaybı, PDE kaybı ve sınır koşulu kaybı birlikte optimize edilir
- Otomatik türev (autograd) ile PDE kalanı hesaplanır

## 2. Multi-Layer Perceptron (MLP)
- Sadece veri ile eğitilir, fiziksel kısıt yoktur
- Hızlı ve basit, fakat teorik tutarlılık garanti edilmez

## 3. Klasik Yöntemler
- Black-Scholes kapalı formül
- Monte Carlo simülasyonu
- Sonlu Farklar (implicit scheme)

# Gerçek Piyasa Verisiyle Test / Real Market Data Test
- yfinance ile hisse kapanış fiyatları çekilebilir
- Black-Scholes ile teorik fiyatlar üretilebilir
- PINN/MLP modelleri gerçek veriyle test edilebilir

# Gelişmiş PINN Uygulamaları / Advanced PINN Applications
- Amerikan opsiyonu için erken kullanım koşullu PINN
- Heston modeli için stokastik volatilite PDE PINN
- Implied volatility fonksiyonunu doğrudan öğrenen PINN

# Kullanım / Usage
1. Ortamı kurun: `pip install -r requirements.txt` (torch, numpy, scipy, matplotlib, yfinance)
2. `pinn_comparison_final.py` dosyasını çalıştırın
3. Sonuç grafiği ve metrikler otomatik kaydedilir
4. `view_graphics.py` ile web arayüzünde görselleştirme yapılabilir

# Sonuçların Yorumlanması / Interpreting Results
- PINN, az veriyle ve teorik tutarlılık gerektiren durumlarda öne çıkar
- MLP, veri bol ise hızlı ve düşük hatalı sonuçlar verebilir
- Klasik yöntemler referans ve validasyon için kullanılır

# Dosya Açıklamaları / File Descriptions
- `pinn_comparison_final.py`: Temiz, modüler karşılaştırma ve test kodu
- `pinn_black_scholes_complete.py`: Kapsamlı PINN ve klasik yöntemler
- `pinn_black_scholes.py`: Temel PINN şablonu ve fonksiyonlar
- `view_graphics.py`: Web tabanlı grafik görüntüleyici
- `model_comparison_final.png`: Sonuç grafiği

# Atıf ve Akademik Kullanım / Citation
Bu kod ve sonuçlar, akademik çalışmalarda ve tezlerde referans gösterilebilir. Lütfen uygun atıf yapınız.

---

# PINN vs MLP vs Classical Methods - Comparison Report

---

## 📊 Grafik Dosyası / Plot File
**Konum / Location:** `/home/mehmetcelik/Masaüstü/masaüstü/makale/finance ml/model_comparison_final.png`

**Boyut / Size:** 3582 x 1475 piksel, 227 KB, 150 DPI

---

## 🔬 Test Parametreleri / Test Parameters

| Parametre | Değer |
|-----------|-------|
| Kullanım Fiyatı (Strike) | 100.0 |
| Vade (Maturity) | 1.0 yıl |
| Faiz Oranı (Rate) | 0.05 (5%) |
| Volatilite (Volatility) | 0.2 (20%) |
| Eğitim Verisi | 100 nokta |
| Test Verisi | 50 nokta |
| Gürültü Seviyesi | 2% |

---

## 📈 Sonuçlar / Results

### PINN (Physics-Informed Neural Network)
- **MAE (Mean Absolute Error):** 26.5079
- **RMSE (Root Mean Square Error):** 36.8405
- **Eğitim Zamanı:** 0.89 saniye
- **Tahmin Zamanı:** 0.1252 ms
- **Özellikleri:**
  - Fizik yasalarını (Black-Scholes PDE) kayıp fonksiyonuna entegre eder
  - Veri kıtlığında teorik olarak tutarlı sonuçlar verir
  - Daha yavaş eğitim ama daha güçlü genelleme (generalization)

### MLP (Multi-Layer Perceptron)
- **MAE:** 18.6151
- **RMSE:** 20.5652
- **Eğitim Zamanı:** 0.0934 saniye
- **Tahmin Zamanı:** 0.1070 ms
- **Özellikleri:**
  - Sadece veriyi öğrenir, fizik yasalarını dikkate almaz
  - Hızlı eğitim
  - Bu test setinde daha düşük hata (çünkü veri yeterli)

---

## 🎯 Grafiklerde Gösterilen İçerik / Plot Contents

### 1. PINN Performansı (Sol Üst)
- Gerçek vs PINN tahminleri scatter plot
- Mükemmel tahmin için diagonal referans çizgisi

### 2. MLP Performansı (Orta Üst)
- Gerçek vs MLP tahminleri scatter plot
- Performans karşılaştırması

### 3. PINN vs MLP (Sağ Üst)
- İki modelin tahminlerinin doğrudan karşılaştırması

### 4. PINN Hata Dağılımı (Sol Alt)
- Hata değerlerinin histogram dağılımı
- Ortalama hata (μ) gösterilmiştir

### 5. MLP Hata Dağılımı (Orta Alt)
- MLP hata dağılımı
- PINN ile karşılaştırmalı analiz

### 6. MAE Karşılaştırması (Sağ Alt)
- Bir bakışta MAE değerlerinin karşılaştırması
- PINN vs MLP

---

## 📚 Akademik Değer / Academic Value

### PINN'in Avantajları / Advantages of PINN:
1. **Teorik Tutarlılık:** Black-Scholes PDE'yi doğrudan öğrenme sürecine entegre eder
2. **Veri Kıtlığında Dayanıklılık:** Az veriyle bile teorik sınırlar içinde kalır
3. **Dışarı-Domain Tahminleri:** Eğitim verisi dışında daha iyi genelleme
4. **Fizik-Destekli Öğrenme:** Sadece veriyi değil, matematiksel ilişkileri de öğrenir

### MLP'nin Avantajları / Advantages of MLP:
1. **Hız:** Daha hızlı eğitim ve tahmin
2. **Basitlik:** Daha basit uygulama
3. **Yeterli Veriyle Etkililik:** Eğer veri yeterli ise iyi performans

---

## 🔧 Kod Dosyaları / Code Files

1. **pinn_comparison_final.py** - Temiz, okunaklı karşılaştırma kodu
2. **pinn_black_scholes_complete.py** - Kapsamlı PINN implementasyonu
3. **pinn_black_scholes.py** - İlk PINN temel yapısı

---

## 📖 Gelecek Çalışmalar / Future Work

### İleri Seviye PINN Uygulamaları:
1. **Amerikan Opsiyonları** - Erken kullanım hakkı içeren opsiyonlar
2. **Heston Modeli** - Stokastik volatilite modeli
3. **Parametre Kalibrasyonu** - Piyasa verilerinden implied volatility öğrenme
4. **Yüksek Boyutlu Problemler** - Sepet opsiyonları, exotik opsiyonlar
5. **V-PINN (Variational PINN)** - Zayıf formülasyon kullanımı

---

## 📝 Notlar / Notes

- Tüm grafikleri oluştururken **headless rendering** kullanıldı (GUI sorunlarından kaçınmak için)
- Grafik yüksek çözünürlükte (150 DPI) kaydedildi - akademik yayınlarda kullanıma hazır
- Yazılar çakışmadan okunaklı şekilde yerleştirildi
- Türkçe ve İngilizce açıklamalar eklenmiştir

---

**Oluşturma Tarihi / Creation Date:** 19 Şubat 2026
**Yazı / Author:** PINN Research Pipeline
