import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="RAG Asistanı", layout="centered")
st.title("🤖 RAG Asistanı (Debug Modu)")

# --- 1. API ANAHTARI ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Lütfen Streamlit Secrets kısmına GROQ_API_KEY ekleyin.")
    st.stop()

# --- 2. YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_path = os.path.join(current_dir, "vector_deposu")
        
        vector_store = FAISS.load_local(
            vector_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Sistem Yükleme Hatası: {e}")
        return None

vector_db = load_resources()
if not vector_db: st.stop()

# --- 3. MODEL ---
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# --- 4. SOHBET GEÇMİŞİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. İŞLEM ---
if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan cevaplıyor
    with st.chat_message("assistant"):
        try:
            # A) Chunkları Getir (Similarity Search)
            docs = vector_db.similarity_search(prompt, k=2)
            
            # --- DEBUG: Ekrana Chunkları Basıyoruz (Cevaptan Önce Görelim) ---
            st.markdown("### 🔍 Yapay Zekanın Okuduğu Parçalar (Chunks):")
            
            context_text = ""
            for i, doc in enumerate(docs):
                # Metadata'yı güvenli çekelim
                source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
                page_label = doc.metadata.get("page", doc.metadata.get("page_label", "-"))
                
                # Chunk İçeriği
                chunk_content = doc.page_content
                
                # Context'e ekle
                context_text += chunk_content + "\n\n"

                # Ekrana Bas (Kutu İçinde)
                st.info(f"**📄 Kaynak {i+1}:** {source_name} (Sayfa: {page_label})\n\n---\n\n{chunk_content}")

            # B) Cevap Üret
            if not context_text.strip():
                st.warning("Veritabanında ilgili bilgi bulunamadı!")
            else:
                messages = [
                    SystemMessage(content=f"Sen bir asistanın. Sadece şu metne bakarak cevap ver:\n{context_text}"),
                    HumanMessage(content=prompt)
                ]
                
                response = llm.invoke(messages)
                
                # C) Cevabı Göster
                st.markdown("### 🤖 Cevap:")
                st.markdown(response.content)
                
                # Geçmişe kaydet
                st.session_state.messages.append({"role": "assistant", "content": response.content})

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
