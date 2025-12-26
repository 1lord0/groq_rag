import os
class BelgeAracı:
    def __init__(self, yol):
        # Burası hazırlık aşaması
        # "yol" bilgisini sınıfa kaydedeceğiz.
        self.klasor_yolu = yol

    def dosyalari_listele(self):
        try:
            dosyalar=os.listdir(self.klasor_yolu)
            for dosya in dosyalar:
                    print(f"bulunan dosyalar: {dosya}")
                    
        
        
        except FileNotFoundError:
            print(f"hata :{self.klasor_yolu} dizini bulunamadı")
        
        except Exception as e:
            print(f"beklenmedik hata oluştu {e}")
            
            
            
            
        # Klasördeki dosyaları ekrana yazdıracağız.
        
        pass # Şimdilik boş bırakıyoruz
    def pdf_sayisi(self):
        try:
            dosyalar=os.listdir(self.klasor_yolu)
            pdf_liste=[]
            for dosya in dosyalar:
                if dosya.endswith(".pdf"):
                    pdf_liste.append(dosya)
            print(f"dosya sayısı:{len(pdf_liste)}")
        except FileNotFoundError:
            print("klasör bulunamadı")
    
    def parcala_ve_hazırla(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader,TextLoader
        docs=[]
        dosyalar=os.listdir(self.klasor_yolu)
        try:
            
            for dosya in dosyalar:
                path = os.path.join(self.klasor_yolu, dosya)
                if dosya.endswith(".pdf"):
                    loader=PyPDFLoader(path)
                    docs.extend(loader.load())
                elif dosya.endswith(".txt"):
                    loader=TextLoader(path)
                    docs.extend(loader.load())
                
                    
                text_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=1200,
                        chunk_overlap=200)
                chunks=text_splitter.split_documents(docs)
                
                print(f"başarıyla  {len(chunks)} chunks oluşturuldu")
                return chunks
        except Exception as e:
            print(f"dosyalar işlenirken bir hata oluştu:{e}")

            return []    
        
    def index_olusturma(self,chunks):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        try:
            embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            db=FAISS.from_documents(chunks, embeddings)
            
            db.save_local("vector_deposu")
            
            print("vetör veri tabanı başaryıla oluşturuldu")
        except Exception as e :
            print(f"index oluşturulurken hata oluştu {e}")
            

    
        
    def cevap_uret(self, soru, kaynaklar):
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        # 1. Kaynak kontrolü
        if not kaynaklar:
            return "Hata: Aranan konuyla ilgili dökümanda bilgi bulunamadı."
    
        try:
            # Modeli tanımla
            llm = ChatGroq(
                temperature=0, 
                model_name="llama-3.1-8b-instant", 
                groq_api_key="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"
            )
            
            # 2. Context oluşturma
            context = "\n\n".join([doc.page_content for doc in kaynaklar])
            
            # 3. Prompt (Sistem Talimatı)
            template = """Sen yardımcı bir asistansın. Aşağıdaki kaynaklara dayanarak soruyu cevapla.
            Eğer cevap kaynaklarda yoksa, uydurma ve 'Bu bilgiye sahip değilim' de.
            
            Kaynaklar: {context}
            Soru: {soru}
            Cevap:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # 4. Zinciri oluştur ve çalıştır
            chain = prompt | llm
            sonuc = chain.invoke({"context": context, "soru": soru})
    
            # 5. None kontrolü (Burada hata olabilir)
            if sonuc and hasattr(sonuc, 'content'):
                return sonuc.content
            else:
                return "Hata: Modelden boş yanıt döndü."
    
        except Exception as e:
            print(f"DEBUG - Hata oluştu: {e}")
            return f"Sistem hatası: {str(e)}"    
    def soru_sor(self,query,k=3):
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        try:
            embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db=FAISS.load_local("vector_deposu", embeddings,allow_dangerous_deserialization=True)
            
            
            #benzerlik araması yap
            
            cevaplar=db.similarity_search(query,k=k)
            
            print(f"{query} sorusu için bulunan kaynaklar:")
            for i,doc in enumerate(cevaplar):
                print(f"\n[kaynak:{i+1}:]")
                
                print(doc.page_content[:300] + "...")
            return cevaplar 
        except Exception as e:
            print(f"Arama sırasında hata oluştu: {e}")
            return []
# --- KULLANIM AŞAMASI ---
if __name__ == "__main__":
    # 1. Sınıfı başlat (Dosyalarının olduğu klasörü yaz)
    isleyici = BelgeAracı(yol=r"C:\Users\eren\Desktop\pyhon\data") 

    # 2. Belgeleri oku ve parçala
    parcalar = isleyici.parcala_ve_hazırla()
    # 3. Vektör veritabanını yap ve kaydet
    isleyici.index_olusturma(parcalar)
    
    
    
    


