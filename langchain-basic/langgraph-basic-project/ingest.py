import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from dotenv import load_dotenv
load_dotenv()

import chunk
import glob
# need to install uv add langchain-community faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# to generate and store embeddings need to install uv add langchain-google-genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DOCS_DIR = "sample_docs"
INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_and_split(docs_dir):
    """Load .txt files and split them into ~500 char chunks with overlap
    """
    chunks = []
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        # Split on double-newline(paragraphs-boundry)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": source}))
                # if single paragraph is already > CHUNK_SIZE, keep it as-is
                current = para
    return chunks

def main():
    print(f"Loading and splitting documents from {DOCS_DIR}...")
    chunks = load_and_split(DOCS_DIR)
    print(f" Split into {len(chunks)} chunks")

    print("Embedding and storing in FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved to {INDEX_DIR}")

    print(f"Done! Index saved to {INDEX_DIR}/")
    print(f" Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    main()