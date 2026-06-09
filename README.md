# 🛒 E-Ticaret Müşteri Harcama Segmentasyonu

Makine öğrenmesi kullanarak e-ticaret müşterilerinin **yüksek** veya **düşük** harcama segmentine ait olup olmadığını tahmin eden uçtan uca bir sınıflandırma projesi.

[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) veri seti üzerinde müşteri bazında özellik mühendisliği yapılır, XGBoost sınıflandırıcı eğitilir ve sonuçlar Streamlit arayüzü üzerinden canlı olarak sunulur.

---

## 🌐 Canlı Demo

Uygulamayı tarayıcınızda denemek için:

**👉 [Streamlit Canlı Demo](https://ml-eticaret-tahmin-dd6qcdkyq5t3t5hzvayevr.streamlit.app/)**

---

## 📋 Proje Açıklaması

Bu proje, perakende satış verilerini müşteri düzeyinde toplayarak her müşterinin harcama profilini analiz eder. Hedef değişken; müşterinin toplam harcamasının veri seti medyanının üzerinde olup olmamasına göre belirlenir:

| Etiket | Anlam |
|--------|-------|
| `1` | Yüksek harcamalı müşteri (medyan üstü) |
| `0` | Düşük harcamalı müşteri (medyan altı) |

Müşteri başına türetilen davranışsal özellikler (sipariş sayısı, ürün çeşitliliği, ülke) kullanılarak **XGBoost** modeli eğitilir. Performans karşılaştırması için **Logistic Regression** modeli de değerlendirilir.

---

## ✨ Özellikler

- **Müşteri bazlı veri toplama:** Fatura, ürün çeşidi, harcama ve ülke bilgilerinin aggregate edilmesi
- **XGBoost sınıflandırıcı:** `n_estimators=200`, `random_state=42` ile eğitilmiş ana model
- **Karşılaştırmalı analiz:** Logistic Regression ile yan yana performans değerlendirmesi
- **Kapsamlı metrikler:** Accuracy, Precision, Recall ve F1-Score hesaplaması
- **Model kalıcılığı:** Eğitilmiş model `joblib` ile `.pkl` formatında kaydedilir
- **Streamlit arayüzü:** Türkçe, kullanıcı dostu tahmin ekranı
- **Canlı tahmin:** Sipariş sayısı, ürün çeşidi ve ülke bilgisiyle anlık segment tahmini

---

## 🛠️ Teknolojiler

| Kategori | Araçlar |
|----------|---------|
| Dil | Python |
| Veri işleme | Pandas, NumPy |
| Makine öğrenmesi | Scikit-learn, XGBoost |
| Model kaydı | Joblib |
| Arayüz | Streamlit |
| Veri formatı | CSV, Excel desteği (openpyxl) |

---

## 📊 Model Performansı

Test seti üzerinde (%20) elde edilen sonuçlar:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **%87.6** | %88.6 | %86.4 | **%87.5** |
| Logistic Regression | %86.2 | %88.3 | %83.5 | %85.8 |

> XGBoost modeli, tüm metriklerde Logistic Regression'a kıyasla daha iyi performans göstermiştir.

---

## 📁 Dosya Yapısı

```
ml_odev/
│
├── online_retail_II.csv    # Ham e-ticaret veri seti
├── model_egit.py           # Veri işleme, model eğitimi ve kayıt
├── app.py                  # Streamlit tahmin uygulaması
├── xgboost_model.pkl       # Eğitilmiş XGBoost modeli (model_egit.py sonrası oluşur)
├── requirements.txt        # Python bağımlılıkları
└── README.md               # Proje dokümantasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu klonlayın

```bash
git clone https://github.com/KULLANICI_ADI/ml_odev.git
cd ml_odev
```

### 2. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 3. Modeli eğitin

```bash
python model_egit.py
```

Bu adım veri setini işler, modelleri eğitir, performans tablosunu terminale yazdırır ve `xgboost_model.pkl` dosyasını oluşturur.

### 4. Streamlit uygulamasını başlatın

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresinde uygulama açılır.

---

## 🎯 Model Girdileri

Streamlit uygulamasında kullanılan özellikler:

| Özellik | Açıklama | Tip |
|---------|----------|-----|
| `sipariş_sayısı` | Müşterinin toplam sipariş (fatura) sayısı | int |
| `ürün_çeşidi` | Satın alınan benzersiz ürün sayısı | int |
| `ülke_encoded` | En sık alışveriş yapılan ülke (sayısal kod) | int |

**Ülke kodları:**

| Ülke | Kod |
|------|-----|
| United Kingdom | 0 |
| Germany | 1 |
| France | 2 |
| EIRE | 3 |
| Spain | 4 |

---

## 📌 Notlar

- `online_retail_II.csv` dosyası proje kök dizininde bulunmalıdır.
- Model eğitimi büyük veri seti nedeniyle birkaç dakika sürebilir.
- `xgboost_model.pkl` dosyası `model_egit.py` çalıştırılmadan önce Streamlit uygulaması hata verecektir.

---

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
