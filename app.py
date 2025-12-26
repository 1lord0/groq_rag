import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# --- AYARLAR VE GÜVENLİK ---
# Streamlit Cloud'da 'Settings > Secrets' kısmına GROQ_API_KEY eklemeyi unutma!
api_key = st.secrets.get("GROQ_API_KEY") 
client = Groq(api_key=api_key)

# --- MODEL VE VERİTABANI YÜKLEME ---
@st.cache_resource # Sayfa her yenilendiğinde modeli tekrar yüklemez, hız kazandırır
def load_system():
    try:
        # 1. Embedding Modelini Yükle
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Vektör Deposunu Yükle (Klasör ismin: vector_deposu)
        vector_db = FAISS.load_local(
            "vector_deposu", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_db
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        return None

# --- CEVAP ÜRETME FONKSİYONU ---
def get_ai_response(user_query, vector_db):
    try:
        # Benzer dökümanları bul
        docs = vector_db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Groq ile cevap üret
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"Sen bir asistansın. Verilen bağlama göre cevap ver: {context}"},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Cevap üretilemedi: {e}"

# --- STREAMLIT ARAYÜZÜ ---
def main():
    st.set_page_config(page_title="RAG Asistanı", page_icon="🤖")
    st.title("🤖 Boran Tarzı RAG Sistemi")
    st.markdown("---")

    vector_db = load_system()

    if vector_db:
        # Sohbet Geçmişi Başlat (Opsiyonel)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Eski mesajları göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Kullanıcıdan soru al
        if prompt := st.chat_input("Döküman hakkında bir şey sorun..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = get_ai_response(prompt, vector_db)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
