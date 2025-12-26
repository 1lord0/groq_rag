from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Embedding modelini tanımla (Bilgisayarındakiyle aynı olmalı)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. PDF okuma/parçalama kısımlarını atla, direkt indeksi yükle
def load_rag_system():
    # GitHub'a yüklediğin klasör ismini buraya yazıyoruz
    vector_store = FAISS.load_local(
        "my_faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True # FAISS yüklemesi için bu şarttır
    )
    return vector_store

# Sistemi başlat
vector_db = load_rag_system()
