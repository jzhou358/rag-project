import os
from dotenv import load_dotenv

import streamlit as st

# These imports match the style used in functions_for_pipeline.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ---------------------------
# 1. Load environment variables
# ---------------------------
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your .env file"


# ---------------------------
# 2. Load the three vector stores (chunks, summaries, quotes)
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
# 3. Define a stronger RAG prompt
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
# 5. A "medium" RAG pipeline using all three retrievers
# ---------------------------
def medium_rag_answer(question: str):
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
# 6. Streamlit UI
# ---------------------------
st.title("Medium RAG over mlq.pdf (Chunks + Summaries + Quotes)")

default_q = "Ask a question about the document (mlq.pdf)..."
user_q = st.text_input("Your question:", value=default_q)

if st.button("Run Medium RAG"):
    q = user_q.strip()
    if not q:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and answering..."):
            answer, used_context = medium_rag_answer(q)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            st.write(used_context)
