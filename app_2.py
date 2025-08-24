import os
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# -------------------------------
# PATCH ASYNC FOR STREAMLIT
# -------------------------------
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "./data"
CHROMA_DIR = "./chroma_db"
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    st.error("🚨 GOOGLE_API_KEY not set. Please set it as an environment variable.")
    st.stop()

# -------------------------------
# LOAD CUAD CSV AS DOCUMENTS
# -------------------------------
def load_csv_as_docs(file_path: str):
    df = pd.read_csv(file_path)
    docs = []
    for i, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col])])
        docs.append(Document(page_content=text, metadata={"row": i}))
    return docs

@st.cache_resource
def load_chroma():
    file_path = os.path.join(DATA_PATH, "master_clauses.csv")
    docs = load_csv_as_docs(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # build / reload
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_DIR)
        vectorstore.persist()

    return vectorstore

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="AI Contract Clause Checker", layout="wide")
    st.title("📜 AI Contract Clause Checker & Negotiator")
    st.markdown("Upload a contract, and let AI flag risky clauses based on CUAD, suggest safer alternatives, and propose negotiation terms.")

    vectorstore = load_chroma()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    uploaded_file = st.file_uploader("📂 Upload your contract (.txt)", type=["txt"])
    if uploaded_file is not None:
        contract_text = uploaded_file.read().decode("utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        contract_chunks = splitter.split_text(contract_text)

        st.subheader("🔍 Clause Analysis")
        for i, chunk in enumerate(contract_chunks):
            st.markdown(f"### Clause {i+1}")
            st.write(chunk)

            # 1️⃣ Retrieve similar CUAD risks
            similar_docs = retriever.get_relevant_documents(chunk)
            cuad_context = "\n\n".join([doc.page_content for doc in similar_docs])

            # Show CUAD matches on UI
            with st.expander(f"📚 Relevant CUAD Risks for Clause {i+1}"):
                for j, doc in enumerate(similar_docs):
                    st.markdown(f"**Match {j+1}:** {doc.page_content}")

            # 2️⃣ Build structured prompt using ChatPromptTemplate
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a contract risk analyzer. Use the CUAD dataset context below to ground your analysis. "
                 "Respond in a professional note style, not bullet points."),
                ("human", 
                 "CUAD Risks Context:\n{cuad_context}\n\n"
                 "Now analyze this clause:\n{clause}\n\n"
                 "Format your answer as a short professional note with these sections (each of lines not exceeding 5) not as paragraphs:\n"
                 "Clause Title:\n"
                 "Risk Assessment: (1. Risky: <Yes/No> 2. Why risky?: <1–5 lines more>) 3.Key Factors to label as risky/Not risky\n"
                 "Safer Alternative:\n"
                 "Negotiation Suggestion:\n")
            ])

            query = prompt_template.format_messages(
                cuad_context=cuad_context,
                clause=chunk
            )

            try:
                result = llm.invoke(query)
                st.markdown("✅ **Analysis Result:**")
                st.write(result.content)
            except Exception as e:
                st.error(f"Error analyzing clause {i+1}: {e}")


if __name__ == "__main__":
    main()
# import os
# import streamlit as st
# import pandas as pd
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# import asyncio
# import nest_asyncio
# from dotenv import load_dotenv

# # -------------------------------
# # PATCH ASYNC FOR STREAMLIT
# # -------------------------------
# nest_asyncio.apply()
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# # -------------------------------
# # CONFIG
# # -------------------------------
# DATA_PATH = "./data"
# CHROMA_DIR = "./chroma_db"
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if GOOGLE_API_KEY is None:
#     st.error("🚨 GOOGLE_API_KEY not set. Please set it as an environment variable.")
#     st.stop()

# # -------------------------------
# # LOAD CUAD CSV AS DOCUMENTS
# # -------------------------------
# def load_csv_as_docs(file_path: str):
#     df = pd.read_csv(file_path)
#     docs = []
#     for i, row in df.iterrows():
#         text = " | ".join([f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col])])
#         docs.append(Document(page_content=text, metadata={"row": i}))
#     return docs

# @st.cache_resource
# def load_chroma():
#     file_path = os.path.join(DATA_PATH, "master_clauses.csv")
#     docs = load_csv_as_docs(file_path)

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     splits = text_splitter.split_documents(docs)

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

#     # build / reload
#     if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
#         vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
#     else:
#         vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_DIR)
#         vectorstore.persist()

#     return vectorstore

# # -------------------------------
# # Streamlit App
# # -------------------------------
# def main():
#     st.set_page_config(page_title="AI Contract Clause Checker", layout="wide")
#     st.title("📜 AI Contract Clause Checker & Negotiator")
#     st.markdown("Upload a contract, and let AI flag risky clauses based on CUAD, suggest safer alternatives, and propose negotiation terms.")

#     vectorstore = load_chroma()
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

#     uploaded_file = st.file_uploader("📂 Upload your contract (.txt)", type=["txt"])
#     if uploaded_file is not None:
#         contract_text = uploaded_file.read().decode("utf-8")
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         contract_chunks = splitter.split_text(contract_text)

#         st.subheader("🔍 Clause Analysis")
#         for i, chunk in enumerate(contract_chunks):
#             st.markdown(f"**Clause {i+1}:**")
#             st.write(chunk)

#             # 1️⃣ Retrieve similar CUAD risks
#             similar_docs = retriever.get_relevant_documents(chunk)
#             cuad_context = "\n\n".join([doc.page_content for doc in similar_docs])

#             # 2️⃣ Build risk-aware prompt
#             query = f"""
# You are a contract risk analyzer. 
# Use the CUAD dataset context below to ground your analysis.

# CUAD Risks Context:
# {cuad_context}

# Now analyze this clause:

# {chunk}

# Answer ONLY in this strict format:

# Clause: <1–2 word title>
# Risky?: <Yes/No>
# Why risky?: <1–2 lines>
# Safer Alternative: <1–2 lines>
# Negotiation Wording: <1–2 lines>
# """

#             try:
#                 result = llm.invoke(query)
#                 st.success(result.content)
#             except Exception as e:
#                 st.error(f"Error analyzing clause {i+1}: {e}")


# if __name__ == "__main__":
#     main()
