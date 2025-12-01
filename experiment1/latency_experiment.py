"""
latency_experiment.py

Measure end-to-end latency for:
  1) GPT-only (no RAG context)
  2) RAG over a PDF (AutoBurst paper or any other PDF)

Results are written to latency_results.csv with columns:
  question, mode, runs, avg_latency_sec

Before running:
  - Put your PDF file on disk, e.g. data/mydocument.pdf
  - Set OPENAI_API_KEY in your environment.
"""

import os
import time
import csv
from dataclasses import dataclass, asdict
from typing import List

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ---------------------------
# Config
# ---------------------------

PDF_PATH = "data/mydocument.pdf"  # <-- change to your actual path
N_RUNS_PER_MODE = 3             # how many times to repeat each question per mode
OUTPUT_CSV = "latency_results.csv"

# Example test questions about the AutoBurst paper
TEST_QUESTIONS: List[str] = [
    "What problem does AutoBurst aim to solve in cloud computing?",
    "How do burstable instances differ from regular instances in terms of CPU credits and cost?",
    "What are the two main components of AutoBurst and what does each one control?",
    "How does the Latency Optimizer use a PD controller to meet a latency SLO?",
    "How does the Resource Estimator decide how many burstable instances to provision?",
    "Against which existing system is AutoBurst evaluated, and what kind of cost savings are reported?",
    "What kinds of real-world traces are used to evaluate AutoBurst's performance?",
    "How does AutoBurst handle mispredictions in the number of instances while still meeting latency SLOs?",
]

# ---------------------------
# Load environment
# ---------------------------

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your environment"

# ---------------------------
# LLM and RAG prompt
# ---------------------------

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

RAG_PROMPT = PromptTemplate.from_template(
    """
You are a careful assistant answering questions based ONLY on the provided context.

Your job:
1. Read all context carefully.
2. Synthesize a precise answer.
3. If the answer is not clearly supported by the context, say:
   "I don't know based on the document."

Use step-by-step reasoning internally, but only output the final answer.

Context:
{context}

Question: {question}

Final answer:
"""
)

# ---------------------------
# RAG building and answer functions
# ---------------------------

def build_pdf_retriever(pdf_path: str):
    """Load a PDF, split it into chunks, build FAISS, and return a retriever."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[INFO] Loading PDF from {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    print(f"[INFO] Building FAISS index with {len(chunks)} chunks ...")
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    return retriever


def rag_answer(retriever, question: str) -> str:
    """Answer using RAG over the retriever."""
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    prompt_text = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)
    return response.content


def gpt_only_answer(question: str) -> str:
    """Answer using GPT only, without any retrieved context."""
    prompt = (
        "You are a knowledgeable assistant. "
        "Answer the following question using your general knowledge only.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content


# ---------------------------
# Latency measurement helpers
# ---------------------------

@dataclass
class LatencyResult:
    question: str
    mode: str          # "gpt_only" or "rag"
    runs: int
    avg_latency_sec: float


def measure_latency(fn, question: str, runs: int) -> float:
    """Call fn(question) multiple times and return average wall-clock latency."""
    latencies = []
    for i in range(runs):
        start = time.perf_counter()
        _ = fn(question)
        end = time.perf_counter()
        lat = end - start
        latencies.append(lat)
        print(f"  Run {i + 1}/{runs}: {lat:.3f} s")
    return sum(latencies) / len(latencies)


# ---------------------------
# Main experiment
# ---------------------------

def main():
    print("[INFO] Building retriever for RAG...")
    retriever = build_pdf_retriever(PDF_PATH)

    results: List[LatencyResult] = []

    for idx, q in enumerate(TEST_QUESTIONS, start=1):
        print("=" * 80)
        print(f"Question {idx}: {q}")

        # GPT-only baseline
        print("[GPT-only] Measuring latency...")
        avg_no_rag = measure_latency(gpt_only_answer, q, N_RUNS_PER_MODE)
        print(f"[GPT-only] Average latency: {avg_no_rag:.3f} s")
        results.append(
            LatencyResult(
                question=q,
                mode="gpt_only",
                runs=N_RUNS_PER_MODE,
                avg_latency_sec=avg_no_rag,
            )
        )

        # RAG
        print("[RAG] Measuring latency...")
        avg_rag = measure_latency(lambda qq: rag_answer(retriever, qq), q, N_RUNS_PER_MODE)
        print(f"[RAG] Average latency: {avg_rag:.3f} s")
        results.append(
            LatencyResult(
                question=q,
                mode="rag",
                runs=N_RUNS_PER_MODE,
                avg_latency_sec=avg_rag,
            )
        )

    # Write CSV
    print("=" * 80)
    print(f"[INFO] Writing results to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "mode", "runs", "avg_latency_sec"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print("[INFO] Done. You can now load latency_results.csv into Excel / pandas and plot.")
    print("Columns: question, mode (gpt_only|rag), runs, avg_latency_sec")


if __name__ == "__main__":
    main()
