import streamlit as st
from belge_araci import BelgeAracı
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Astroloji & Belge Asistanı",
    page_icon="🔮",
    layout="centered"
)

# --- STYLES (Opsiyonel Görsel Dokunuş) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("🔮 AI Belge Asistanı")
st.info("Dökümanlarınızdaki bilgilere dayanarak sorularınızı cevaplarım.")

# --- BELGE İŞLEYİCİSİNİ BAŞLAT ---
# GitHub'daki klasör yapına göre 'data' klasörünü hedefliyoruz.
# Streamlit Cloud'da çalışırken yolun doğru olduğundan emin oluyoruz.
data_yolu = "data" 
isleyici = BelgeAracı(yol=data_yolu)

# --- CHAT GEÇMİŞİ (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SORU-CEVAP AKIŞI ---
if prompt := st.chat_input("Dökümanlarla ilgili bir şey sorun..."):
    
    # 1. Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Yanıt üretme aşaması
    with st.chat_message("assistant"):
        with st.spinner("Belgeler taranıyor ve cevap hazırlanıyor..."):
            try:
                # Vektör veritabanından en alakalı 5 parçayı getir (k=5 yaptık)
                kaynaklar = isleyici.soru_sor(prompt, k=5)
                
                # Streamlit Cloud üzerindeki Secrets'tan API anahtarını çek
                # Yerelde çalışırken .streamlit/secrets.toml dosyasında olmalı
                api_key = st.secrets["GROQ_API_KEY"]
                
                # Cevabı üret
                cevap = isleyici.cevap_uret(prompt, kaynaklar, api_key)
                
                # Cevabı ekrana bas
                st.markdown(cevap)
                
                # Kaynakları gösteren açılır menü (isteğe bağlı)
                with st.expander("🔍 Hangi kaynaklara bakıldı?"):
                    for i, doc in enumerate(kaynaklar):
                        st.write(f"**Parça {i+1}:** {doc.page_content[:200]}...")
                
                # Hafızaya kaydet
                st.session_state.messages.append({"role": "assistant", "content": cevap})
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {str(e)}")
                st.warning("Not: Streamlit Secrets ayarlarında GROQ_API_KEY tanımlı olduğundan emin olun.")

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("Sistem Bilgisi")
    st.write("📂 **Veri Klasörü:** `data/`")
    st.write("🗄️ **Vektör Deposu:** `vector_deposu/` (Hazır Yüklendi)")
    
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()