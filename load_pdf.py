import os
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_from_multiple_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(path)
            metadata = {"source": filename}
            documents.append(Document(page_content=text, metadata=metadata))
    return documents
  
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def save_to_faiss(chunks, faiss_path="./faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(faiss_path)
    print(f"✅ FAISS 저장 완료: {faiss_path}")
    return vectorstore


pdf_folder_path = "./pdf"  # PDF 폴더 경로
raw_docs = extract_from_multiple_pdfs(pdf_folder_path)
chunked_docs = split_documents(raw_docs)
faiss_index = save_to_faiss(chunked_docs)