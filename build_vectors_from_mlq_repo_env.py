import os
from dotenv import load_dotenv

# Load .env
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing in .env"

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Path to your PDF (put mlq.pdf in the repo root)
PDF_PATH = "mlq.pdf"   # or "data/mlq.pdf" if you put it in a subfolder

assert os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

print("Creating embeddings with OpenAIEmbeddings...")
embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY

# Build and save vector stores
print("Building chunks_vector_store...")
chunks_vector_store = FAISS.from_documents(chunks, embeddings)
chunks_vector_store.save_local("chunks_vector_store")

print("Building chapter_summaries_vector_store...")
chapter_summaries_vector_store = FAISS.from_documents(chunks, embeddings)
chapter_summaries_vector_store.save_local("chapter_summaries_vector_store")

print("Building book_quotes_vectorstore...")
book_quotes_vectorstore = FAISS.from_documents(chunks, embeddings)
book_quotes_vectorstore.save_local("book_quotes_vectorstore")

print("Done! Vector stores saved in:")
print("  - chunks_vector_store/")
print("  - chapter_summaries_vector_store/")
print("  - book_quotes_vectorstore/")
