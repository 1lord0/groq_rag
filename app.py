import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="RAG Debugger", layout="centered")
st.title("🛠️ RAG Debug Modu: Chunk Kontrolü")
st.info("Bu modda, yapay zekanın okuduğu metinleri (chunk) cevaptan önce görebilirsin.")

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
        # Geçmişte kullanılan chunkları da göster (Gri kutuda)
        if "chunks" in message:
            with st.expander("🔍 Bu cevap için okunan metinleri gör"):
                for chunk in message["chunks"]:
                    st.caption(f"📄 {chunk['source']} (Sayfa: {chunk['page']})")
                    st.text(chunk['content']) # Metni ham haliyle basar
                    st.divider()

# --- 5. İŞLEM ---
if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan cevaplıyor
    with st.chat_message("assistant"):
        try:
            # A) Chunkları Getir
            docs = vector_db.similarity_search(prompt, k=2)
            
            context_text = ""
            found_chunks = []

            # --- DEBUG: BULUNAN METİNLERİ CANLI GÖSTER ---
            st.markdown("##### 🔎 Yapay Zeka Şu Metinleri Okuyor:")
            
            for i, doc in enumerate(docs):
                # Dosya adını temizle
                full_path = doc.metadata.get("source", "Bilinmiyor")
                file_name = os.path.basename(full_path)
                page_label = doc.metadata.get("page", doc.metadata.get("page_label", "-"))
                
                # Metni al
                chunk_content = doc.page_content
                
                # Ekrana bas (Kullanıcı burada kontrol edecek)
                with st.container(border=True):
                    st.markdown(f"**Parça {i+1}** - *{file_name} (Sayfa: {page_label})*")
                    st.code(chunk_content, language="text") # Metni kod bloğu gibi net gösterir
                
                # Listeye ekle
                context_text += chunk_content + "\n\n"
                found_chunks.append({
                    "source": file_name,
                    "page": page_label,
                    "content": chunk_content
                })

            # B) Cevap Üret
            if not context_text.strip():
                st.warning("⚠️ Veritabanında bu soruyla ilgili hiçbir metin bulunamadı!")
                response_text = "Veri tabanında bilgi yok."
            else:
                messages = [
                    SystemMessage(content=f"Sen bir asistanın. Sadece sana verilen şu metne sadık kalarak cevap ver:\n{context_text}"),
                    HumanMessage(content=prompt)
                ]
                
                with st.spinner("Cevap hazırlanıyor..."):
                    response = llm.invoke(messages)
                    response_text = response.content
                
                # C) Cevabı Göster
                st.markdown("### 💡 Cevap:")
                st.markdown(response_text)
                
                # Geçmişe kaydet (Chunklarla beraber)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "chunks": found_chunks
                })

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
