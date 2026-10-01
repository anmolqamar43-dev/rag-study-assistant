import os
import glob
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# 1. Load environment variables
load_dotenv()

# 2. CREATE FASTAPI APP FIRST
app = FastAPI(title="AI Study Assistant using RAG")

# 3. Environment & Directory Setup (Vercel Fix Included)
if os.getenv("VERCEL"):
    DB_DIR = "/tmp/db"
    PDF_DIR = "/tmp/pdfs"
else:
    DB_DIR = "./db"
    PDF_DIR = "./pdfs"

os.makedirs("static", exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. ChromaDB & Groq Setup
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="python_notes",
    embedding_function=embedding_fn
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class QuestionRequest(BaseModel):
    question: str

def custom_text_splitter(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

# 5. API Routes
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process_pdfs")
def process_pdfs():
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        raise HTTPException(
            status_code=400, 
            detail="No PDF files found in 'pdfs/' directory. Please add your 5 PDF files."
        )

    all_chunks = []
    metadatas = []
    ids = []
    chunk_counter = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        
        pdf_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"

        chunks = custom_text_splitter(pdf_text, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            chunk_counter += 1
            all_chunks.append(chunk)
            metadatas.append({"source": filename, "chunk_id": i})
            ids.append(f"doc_{chunk_counter}")

    if all_chunks:
        collection.upsert(
            documents=all_chunks,
            metadatas=metadatas,
            ids=ids
        )

    return {
        "message": f"Successfully processed {len(pdf_files)} PDFs and created {len(all_chunks)} embeddings in ChromaDB!"
    }

@app.post("/ask")
def ask_question(req: QuestionRequest):
    if not groq_client:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY is missing. Please set it in .env file."
        )

    results = collection.query(
        query_texts=[req.question],
        n_results=3
    )

    retrieved_docs = results['documents'][0] if results['documents'] else []
    context = "\n\n".join(retrieved_docs) if retrieved_docs else "No context found."

    system_prompt = (
        "You are an AI Study Assistant for Python programming.\n"
        "Answer the user's question accurately using ONLY the provided context below.\n\n"
        f"Context:\n{context}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.question}
            ],
            temperature=0.2
        )
        answer = completion.choices[0].message.content
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))