import os
import tempfile

from dotenv import load_dotenv
import streamlit as st

# LangChain / OpenAI imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# For uploaded-PDF mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------
# 1. Load environment variables
# ---------------------------
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your .env file"


# ---------------------------
# 2. Load the prebuilt vector stores (chunks, summaries, quotes)
#    These indexes are generated offline from a reference document.
# ---------------------------
@st.cache_resource
def load_retrievers():
    embeddings = OpenAIEmbeddings()

    # Base path is the working directory inside the container (/app)
    chunks_vs = FAISS.load_local(
        "chunks_vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    summaries_vs = FAISS.load_local(
        "chapter_summaries_vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    quotes_vs = FAISS.load_local(
        "book_quotes_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    chunks_ret = chunks_vs.as_retriever(search_kwargs={"k": 3})
    summaries_ret = summaries_vs.as_retriever(search_kwargs={"k": 2})
    quotes_ret = quotes_vs.as_retriever(search_kwargs={"k": 5})

    return chunks_ret, summaries_ret, quotes_ret


chunks_retriever, summaries_retriever, quotes_retriever = load_retrievers()


# ---------------------------
# 3. Define a RAG prompt
# ---------------------------
RAG_PROMPT = PromptTemplate.from_template(
    """
You are a careful assistant answering questions based ONLY on the provided context.

You have context from:
- Detailed chunks from the document
- Chapter-level summaries
- Key quotes

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
# 4. Initialize LLM
# ---------------------------
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


# ---------------------------
# 5. RAG pipeline using the three prebuilt retrievers
# ---------------------------
def medium_rag_answer(question: str):
    """RAG over the built-in reference document (precomputed vector stores)."""
    # 5.1 Retrieve from each store
    docs_chunks = chunks_retriever.get_relevant_documents(question)
    docs_summaries = summaries_retriever.get_relevant_documents(question)
    docs_quotes = quotes_retriever.get_relevant_documents(question)

    # 5.2 Build labeled context
    def fmt_block(label, docs):
        if not docs:
            return ""
        joined = "\n\n".join(d.page_content for d in docs)
        return f"=== {label} ===\n{joined}\n\n"

    context = ""
    context += fmt_block("CHUNKS", docs_chunks)
    context += fmt_block("SUMMARIES", docs_summaries)
    context += fmt_block("QUOTES", docs_quotes)

    # 5.3 Fill the prompt
    prompt_text = RAG_PROMPT.format(context=context, question=question)

    # 5.4 Call the LLM
    response = llm.invoke(prompt_text)

    return response.content, context


# ---------------------------
# 6. Generic RAG for any retriever (for uploaded PDFs)
# ---------------------------
def rag_answer_with_retriever(retriever, question: str):
    """Generic RAG using any retriever (e.g., for an uploaded PDF)."""
    docs = retriever.get_relevant_documents(question)

    if not docs:
        context = ""
    else:
        context = "\n\n".join(d.page_content for d in docs)

    prompt_text = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)
    return response.content, context


# ---------------------------
# 7. Build a retriever from an uploaded PDF (session-local only)
# ---------------------------
@st.cache_resource(show_spinner="Building vector index from uploaded PDF...")
def build_pdf_retriever(file_bytes: bytes):
    """
    Build a FAISS retriever from an uploaded PDF file.
    The index is kept only for the current Streamlit session.
    """
    # 1. Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 2. Load PDF pages
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # 4. Build FAISS vectorstore in memory
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)

    # 5. Return a retriever
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    return retriever


# ---------------------------
# 8. Streamlit UI
# ---------------------------
st.title("COSC 6376 Cloud Computing - Fall 2025 - Final Project")
st.header("Junchao Zhou - 2401060")
st.subheader("Deployment and Optimization of RAG-Enhanced LLM Agent via DevOps Pipeline")

# Architecture image above the controls
st.image("assets/overall_pipeline.png", use_column_width=True)

# Two modes: built-in RAG vs. upload-your-own PDF RAG
mode = st.radio(
    "Choose RAG mode:",
    ["RAG over built-in document", "RAG over uploaded PDF"],
)

# ---------- Mode 1: built-in reference document ----------
if mode == "RAG over built-in document":
    user_q = st.text_input("Ask any question about the built-in document:")

    if st.button("Run Medium RAG", key="builtin_button"):
        q = user_q.strip()
        if not q:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and answering from the built-in document..."):
                answer, used_context = medium_rag_answer(q)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show retrieved context"):
                st.write(used_context)

# ---------- Mode 2: upload PDF and run RAG ----------
else:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

        # Build / cache retriever
        retriever = build_pdf_retriever(uploaded_file.getvalue())

        user_q = st.text_input("Ask a question about the uploaded PDF:")

        if st.button("Run RAG on uploaded PDF", key="upload_button"):
            q = user_q.strip()
            if not q:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Building answer from your PDF..."):
                    answer, used_context = rag_answer_with_retriever(retriever, q)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Show retrieved context"):
                    st.write(used_context)
    else:
        st.info("Please upload a PDF to start.")

# test build trigger 8
