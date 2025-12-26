import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="RAG Asistanı", layout="centered")
st.title("🤖 Boran Tarzı RAG Asistanı")

# --- 1. API ANAHTARI KONTROLÜ ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ_API_KEY bulunamadı! Lütfen Streamlit Secrets kısmına ekleyin.")
    st.stop()

# --- 2. MODEL VE VEKTÖR DEPOSUNU YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        # Embedding Modeli
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Dosya yolunu tam bulmak için
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_path = os.path.join(current_dir, "vector_deposu")
        
        # Vektör Veritabanını Yükle
        vector_store = FAISS.load_local(
            vector_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
        return None

# Kaynakları yükle
vector_db = load_resources()

if not vector_db:
    st.stop()

# --- 3. LLM (GROQ) AYARI ---
llm = ChatGroq(
    groq_api_key=api_key, 
    model_name="llama-3.1-8b-instant"
)

# --- 4. SOHBET ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Eğer bu mesajda kaynak bilgisi saklanmışsa onu da göster (Opsiyonel)
        if "sources" in message:
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# --- 5. SORU CEVAP ALANI ---
if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cevap Üretimi
    with st.chat_message("assistant"):
        try:
            # A) Benzer dökümanları bul
            docs = vector_db.similarity_search(prompt, k=2)
            
            if not docs:
                st.warning("Bu konuyla ilgili dökümanda bilgi bulunamadı.")
                st.stop()
                
            context = "\n".join([doc.page_content for doc in docs])
            
            # --- YENİ EKLENEN KISIM: KAYNAKLARI ÇEK ---
            source_list = []
            for doc in docs:
                # Metadata içinden dosya yolunu al
                full_source = doc.metadata.get("source", "Bilinmeyen Kaynak")
                # Sadece dosya
