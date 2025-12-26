import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class BelgeAracı:
    def __init__(self, yol):
        self.klasor_yolu = yol
        # Embeddings modelini bir kez başlatıyoruz ki her seferinde yüklenmesin
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def parcala_ve_hazırla(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        
        docs = []
        if not os.path.exists(self.klasor_yolu):
            return []

        dosyalar = os.listdir(self.klasor_yolu)
        for dosya in dosyalar:
            path = os.path.join(self.klasor_yolu, dosya)
            try:
                if dosya.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                elif dosya.endswith(".txt"):
                    loader = TextLoader(path)
                    docs.extend(loader.load())
            except Exception as e:
                print(f"{dosya} okunurken hata oluştu: {e}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, # Senin istediğin büyük parçalar
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Başarıyla {len(chunks)} parça oluşturuldu.")
        return chunks

    def index_olusturma(self, parcalar):
        """Vektör veritabanını oluşturur ve yerel klasöre kaydeder."""
        if not parcalar:
            return
        db = FAISS.from_documents(parcalar, self.embeddings)
        db.save_local("vector_deposu")
        print("Vektör veritabanı 'vector_deposu' klasörüne kaydedildi.")

    def soru_sor(self, query, k=5): # k=5 yaparak daha çok metin getiriyoruz
        try:
            # Vektör deposunu yükle
            db = FAISS.load_local(
                "vector_deposu", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            # Benzerlik araması yap
            return db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Arama hatası: {e}")
            return []

    def cevap_uret(self, soru, kaynaklar, api_key):
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        if not kaynaklar:
            return "Hata: Bilgi bulunamadı."

        try:
            # API anahtarını dışarıdan (Streamlit'ten) alıyoruz
            llm = ChatGroq(
                temperature=0, 
                model_name="llama-3.1-8b-instant", 
                groq_api_key=api_key
            )
            
            context = "\n\n".join([doc.page_content for doc in kaynaklar])
            template = """Sen yardımcı bir asistansın. Kaynaklara dayanarak soruyu cevapla.
            Kaynaklar: {context}
            Soru: {soru}
            Cevap:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm
            sonuc = chain.invoke({"context": context, "soru": soru})
            return sonuc.content
        except Exception as e:
            return f"Sistem hatası: {str(e)}"