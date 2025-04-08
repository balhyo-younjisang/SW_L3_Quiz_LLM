from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_index = FAISS.load_local("./faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

query = "UI 테스트하는 과정을 알려줘"
results = faiss_index.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n🔎 Top {i+1} 결과 (출처: {doc.metadata.get('source')}):")
    print(doc.page_content[:500])  # 처음 500자만 출력
