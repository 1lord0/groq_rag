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
        
        # Dosya yolunu tam bulmak için (Hata almamak adına)
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
    st.stop() # Veritabanı yoksa uygulamayı durdur

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

# --- 5. SORU CEVAP ALANI ---
# DİKKAT: Aşağıdaki kodların hepsi 'if' bloğunun içinde olmalı!
if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını ekle ve göster
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
            
            # B) Mesajları hazırla (Liste formatında)
            messages = [
                SystemMessage(content=f"Sen yardımcı bir asistansın. Aşağıdaki bağlama göre cevap ver:\n\n{context}"),
                HumanMessage(content=prompt)
            ]
            
            # C) Groq'a gönder
            response = llm.invoke(messages)
            
            # D) Cevabı yazdır
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

