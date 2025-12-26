import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Sayfa Ayarları
st.set_page_config(page_title="RAG Asistanı")
st.title("RAG Asistanı")

# 1. API Anahtarı Ayarı (Streamlit Secrets'tan çeker)
# Eğer lokalde çalışıyorsan buraya direkt string olarak yazabilirsin test için: api_key = "gsk_..."
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ_API_KEY bulunamadı! Lütfen Streamlit Secrets kısmına ekleyin.")
    st.stop()

# 2. Modeli ve Vektör Deposunu Yükle (Cache kullanarak hızlandırıyoruz)
@st.cache_resource
def load_resources():
    # Embedding Modeli
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vektör Veritabanı (Senin klasör ismin: vector_deposu)
    vector_store = FAISS.load_local(
        "vector_deposu", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_store

# 3. Yüklemeyi Başlat
try:
    vector_db = load_resources()
except Exception as e:
    st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
    st.stop()

# 4. LLM (Groq) Ayarı
llm = ChatGroq(
    groq_api_key=api_key, 
    model_name="llama3-8b-8192"
)

# 5. Sohbet Arayüzü
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Cevap Üretimi
    with st.chat_message("assistant"):
        # Benzer dökümanları bul
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Modele Prompt Ver
        system_prompt = f"Aşağıdaki bağlama göre soruyu cevapla:\n\nBağlam: {context}\n\nSoru: {prompt}"
        response = llm.invoke(system_prompt)
        
        st.write(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
