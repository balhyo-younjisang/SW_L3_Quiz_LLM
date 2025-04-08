import os
import time
import logging
import pytesseract
import fitz  # PyMuPDF
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ─── 설정 및 로깅 ─────────────────────────────────────
load_dotenv()
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdf")
FAISS_PATH = os.getenv("FAISS_PATH", "./faiss_index")
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
PAGES_TO_SKIP = 13

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ─── PDF → 텍스트 추출 ───────────────────────────────────
def extract_text_from_pdf(pdf_path: str, ocr: bool = False) -> str:
    """
    처음 13페이지는 목차이므로 건너뛰고, 그 이후 페이지에서만 텍스트를 추출합니다.
    빈 페이지는 OCR 처리(optional).
    """
    text_chunks = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        for page_number in range(PAGES_TO_SKIP, total_pages):
            page = doc.load_page(page_number)
            txt = page.get_text().strip()
            if not txt and ocr:
                pix = page.get_pixmap()
                img = pix.get_pil_image()
                txt = pytesseract.image_to_string(img)
            text_chunks.append(txt)
        return "\n".join(text_chunks)
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path}: {e}")
        return ""


def extract_from_multiple_pdfs(
    folder_path: str,
    ocr: bool = False,
    max_workers: int = 4
) -> list[Document]:
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]
    documents = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(extract_text_from_pdf, path, ocr): path
            for path in pdf_files
        }
        for future in as_completed(futures):
            path = futures[future]
            text = future.result()
            if text:
                metadata = {
                    "source": os.path.basename(path),
                    "pages": fitz.open(path).page_count,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                documents.append(Document(page_content=text, metadata=metadata))
                logger.info(f"Extracted {metadata['source']} (skipped first {PAGES_TO_SKIP} pages)")
    logger.info(f"Total PDFs processed: {len(documents)} in {time.time() - start:.1f}s")
    return documents



# ─── 청크 분할 ───────────────────────────────────────────
def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─── FAISS 저장 ─────────────────────────────────────────
def save_to_faiss(
    chunks: list[Document],
    faiss_path: str,
    embedding_model=None
) -> FAISS:
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(faiss_path, exist_ok=True)
    vectorstore.save_local(faiss_path)
    logger.info(f"✅ FAISS index saved at: {faiss_path}")
    return vectorstore


# ─── 메인 실행 ───────────────────────────────────────────
if __name__ == "__main__":
    docs = extract_from_multiple_pdfs(PDF_FOLDER, ocr=OCR_ENABLED, max_workers=MAX_WORKERS)
    if not docs:
        logger.warning("No documents extracted; exiting.")
        exit(1)

    chunks = split_documents(
        docs,
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100"))
    )

    save_to_faiss(
        chunks,
        faiss_path=FAISS_PATH,
        embedding_model=None  # or pass custom model
    )
