import streamlit as st
import pandas as pd
import joblib

# Sayfa yapılandırması
st.set_page_config(
    page_title="Müşteri Harcama Tahmin Uygulaması",
    page_icon="🛒",
    layout="wide",
)

# Kayıtlı modeli yükle
@st.cache_resource
def modeli_yukle():
    return joblib.load("xgboost_model.pkl")

model = modeli_yukle()

# Ülke → sayı eşlemesi
ULKE_MAP = {
    "United Kingdom": 0,
    "Germany": 1,
    "France": 2,
    "EIRE": 3,
    "Spain": 4,
}

st.title("🛒 Müşteri Harcama Segmentasyonu")
st.markdown(
    "Eğitilmiş XGBoost modeli ile müşterinin **yüksek** veya **düşük** harcama "
    "segmentinde olup olmadığını tahmin edin."
)

# Sidebar — kullanıcı girdileri
st.sidebar.header("Müşteri Bilgileri")

siparis_sayisi = st.sidebar.number_input(
    "Toplam Sipariş Sayısı",
    min_value=1,
    value=5,
    step=1,
)

urun_cesidi = st.sidebar.number_input(
    "Toplam Ürün Çeşidi",
    min_value=1,
    value=10,
    step=1,
)

ulke = st.sidebar.selectbox(
    "Ülke",
    options=list(ULKE_MAP.keys()),
)

tahmin_yap = st.sidebar.button("Tahmin Yap", type="primary")

# Ana alan
if tahmin_yap:
    ulke_encoded = ULKE_MAP[ulke]

    # Model girdisi: sipariş_sayısı, ürün_çeşidi, ülke_encoded
    ozellikler = pd.DataFrame(
        [[siparis_sayisi, urun_cesidi, ulke_encoded]],
        columns=["toplam_siparis_sayisi", "toplam_urun_cesidi", "en_cok_ulke_encoded"],
    )

    tahmin = model.predict(ozellikler)[0]
    olasiliklar = model.predict_proba(ozellikler)[0]
    tahmin_olasiligi = olasiliklar[tahmin] * 100

    st.subheader("Tahmin Sonucu")

    if tahmin == 1:
        st.success("**Yüksek Harcamalı Müşteri** 🟢")
    else:
        st.error("**Düşük Harcamalı Müşteri** 🔴")

    st.metric(
        label="Tahmin Olasılığı",
        value=f"%{tahmin_olasiligi:.1f}",
    )

    st.info(
        f"Girilen değerler — Sipariş: **{siparis_sayisi}**, "
        f"Ürün çeşidi: **{urun_cesidi}**, Ülke: **{ulke}**"
    )
else:
    st.info("Sol menüden müşteri bilgilerini girip **Tahmin Yap** butonuna basın.")
