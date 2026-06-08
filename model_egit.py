import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

# Veri setini yükle
df = pd.read_csv("online_retail_II.csv")

# Müşteri ID'si olmayan kayıtları çıkar
df = df.dropna(subset=["Customer ID"])

# Satır bazında harcama tutarını hesapla (adet x birim fiyat)
df["Harcama"] = df["Quantity"] * df["Price"]

# Müşteri ID bazında özellikleri topla
musteri_df = df.groupby("Customer ID").agg(
    toplam_siparis_sayisi=("Invoice", "nunique"),       # Benzersiz fatura sayısı
    toplam_urun_cesidi=("StockCode", "nunique"),        # Benzersiz ürün sayısı
    toplam_harcama=("Harcama", "sum"),                  # Toplam harcama
    en_cok_ulke=("Country", lambda x: x.mode().iloc[0]) # En sık alışveriş yapılan ülke
).reset_index()

# Hedef değişken: medyan üstü harcama = 1, altı = 0
medyan_harcama = musteri_df["toplam_harcama"].median()
musteri_df["hedef"] = (musteri_df["toplam_harcama"] > medyan_harcama).astype(int)

# Kategorik değişkeni LabelEncoder ile sayısala dönüştür
le = LabelEncoder()
musteri_df["en_cok_ulke_encoded"] = le.fit_transform(musteri_df["en_cok_ulke"])

# Model girdileri (hedef ve ham harcama özelliği hariç — veri sızıntısını önlemek için)
ozellikler = ["toplam_siparis_sayisi", "toplam_urun_cesidi", "en_cok_ulke_encoded"]
X = musteri_df[ozellikler]
y = musteri_df["hedef"]

# Veriyi %80 eğitim, %20 test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ana model: XGBoost sınıflandırıcı
xgb_model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_tahmin = xgb_model.predict(X_test)

# Karşılaştırma modeli: Lojistik Regresyon
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_tahmin = lr_model.predict(X_test)

# Her iki model için performans metriklerini hesapla
def metrikleri_hesapla(y_gercek, y_tahmin):
    return {
        "Accuracy": accuracy_score(y_gercek, y_tahmin),
        "Precision": precision_score(y_gercek, y_tahmin),
        "Recall": recall_score(y_gercek, y_tahmin),
        "F1-Score": f1_score(y_gercek, y_tahmin),
    }

xgb_metrikler = metrikleri_hesapla(y_test, xgb_tahmin)
lr_metrikler = metrikleri_hesapla(y_test, lr_tahmin)

# Sonuçları tablo olarak yazdır
sonuc_tablosu = pd.DataFrame([xgb_metrikler, lr_metrikler], index=["XGBoost", "LogisticRegression"])
print("\nModel Performans Karşılaştırması:")
print(sonuc_tablosu.round(4).to_string())

# XGBoost modelini kaydet
joblib.dump(xgb_model, "xgboost_model.pkl")
print("\nXGBoost modeli 'xgboost_model.pkl' olarak kaydedildi.")
