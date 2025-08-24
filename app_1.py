import os
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
import asyncio
import nest_asyncio

# Patch asyncio to allow nested loops (needed for Streamlit + gRPC async)
nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------
# CONFIG
# -------------------------------
# DATA_PATH = "./data/master_clauses.csv"
DATA_PATH = "./data"

CHROMA_DIR = "./chroma_db"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # export GOOGLE_API_KEY="your-key"

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if GOOGLE_API_KEY is None:
    st.error("🚨 GOOGLE_API_KEY not set. Please set it as an environment variable.")
    st.stop()

# -------------------------------
# Load CUAD dataset & store in Chroma
# -------------------------------

import pandas as pd
from langchain.schema import Document

def load_csv_as_docs(file_path: str):
    df = pd.read_csv(file_path)
    
    docs = []
    for i, row in df.iterrows():
        # Convert entire row into a string
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

    # Show progress in UI
    progress = st.progress(0, text="🔄 Creating embeddings...")
    total = len(splits)
    batch_size = 25
    embedded = []

    for i in range(0, total, batch_size):
        batch = splits[i:i+batch_size]
        vectordb = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_DIR)

        embedded.extend(batch)
        progress.progress(min((i+batch_size)/total, 1.0),
                          text=f"Embedded {min(i+batch_size, total)}/{total}")

    progress.empty()
    st.success("✅ Vectorstore ready!")

    # Re-open DB so it's consistent
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore
# def load_chroma():
#     # file_path = os.path.join(DATA_PATH, "master_clauses.csv")
#     file_path = os.path.join(DATA_PATH, "master_clauses.csv")
    
#     docs = load_csv_as_docs(file_path)

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     splits = text_splitter.split_documents(docs)

#     vectorstore = Chroma.from_documents(splits, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
#     return vectorstore

# def load_chroma():
    # file_path = "data/master_clauses.csv"
#     df = pd.read_csv(file_path)

#     # Try to find a column that has textual data
#     text_column = None
#     for col in df.columns:
#         if df[col].dtype == object:  # likely text
#             text_column = col
#             break

#     if text_column is None:
#         raise ValueError("No suitable text column found in CSV.")

#     print(f"Using column: {text_column} for document text")

#     # Load using detected column
#     # loader = CSVLoader(file_path=file_path, source_column=text_column)
    
#     loader = CSVLoader(file_path=file_path, source_column=str(text_column))
#     docs = loader.load()

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
#     vectordb.persist()
#     return vectordb
# def load_chroma():
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001", google_api_key=GOOGLE_API_KEY
#     )

#     if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
#         # Load existing DB
#         vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
#     else:
#         # First-time build
#         df = pd.read_csv(DATA_PATH)
#         texts = df["text"].dropna().tolist()  # CUAD master clauses field is 'text'
#         docs = [Document(page_content=t) for t in texts]

#         vectorstore = Chroma.from_documents(
#             documents=docs,
#             embedding=embeddings,
#             persist_directory=CHROMA_DIR
#         )
#         vectorstore.persist()

#     return vectorstore


# -------------------------------
# Build LangChain RAG pipeline
# -------------------------------
def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return chain


# -------------------------------
# Streamlit App UI
# -------------------------------
def main():
    st.set_page_config(page_title="AI Contract Clause Checker", layout="wide")
    st.title("📜 AI Contract Clause Checker & Negotiator")
    st.markdown("Upload a contract, and let AI flag risky clauses, suggest safer alternatives, and propose negotiation terms.")
    print("Working on Vectorstore ")
    vectorstore = load_chroma()
    print("Vectorstore loaded with", vectorstore._collection.count(), "vectors")
    qa_chain = build_qa_chain(vectorstore)

    uploaded_file = st.file_uploader("📂 Upload your contract (.txt)", type=["txt"])
    if uploaded_file is not None:
        contract_text = uploaded_file.read().decode("utf-8")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        contract_chunks = splitter.split_text(contract_text)

        st.subheader("🔍 Clause Analysis")
        for i, chunk in enumerate(contract_chunks):
            st.markdown(f"**Clause {i+1}:**")
            st.write(chunk)

            query = query = f"""
                You are a contract risk analyzer. 
                Analyze the following clause and answer ONLY in this strict format (keep it short):

                Clause: <1–2 word title>
                Risky?: <Yes/No>
                Why risky?: <1–2 lines>
                Safer Alternative: <1–2 lines>
                Negotiation Wording: <1–2 lines>

                Clause:
                {chunk}
                """
            try:
                result = qa_chain.run(query)
                st.success(result)
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
# from langchain.chains import RetrievalQA

# # -------------------------------
# # CONFIG
# # -------------------------------
# DATA_PATH = "./data/master_clauses.csv"
# CHROMA_DIR = "./chroma_db"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # export GOOGLE_API_KEY="your-key"

# # -------------------------------
# # Load CUAD dataset & store in Chroma
# # -------------------------------
# @st.cache_resource
# def load_chroma():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

#     if os.path.exists(CHROMA_DIR):
#         vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
#     else:
#         df = pd.read_csv(DATA_PATH)
#         texts = df["clause_text"].dropna().tolist()  # main CUAD text field

#         docs = [Document(page_content=t) for t in texts]

#         vectorstore = Chroma.from_documents(
#             documents=docs,
#             embedding=embeddings,
#             persist_directory=CHROMA_DIR
#         )
#         vectorstore.persist()

#     return vectorstore


# # -------------------------------
# # Build LangChain RAG pipeline
# # -------------------------------
# def build_qa_chain(vectorstore):
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff"
#     )
#     return chain


# # -------------------------------
# # Streamlit App UI
# # -------------------------------
# def main():
#     st.set_page_config(page_title="AI Contract Clause Checker", layout="wide")
#     st.title("📜 AI Contract Clause Checker & Negotiator")
#     st.markdown("Upload a contract, and let AI flag risky clauses, suggest safer alternatives, and propose negotiation terms.")

#     vectorstore = load_chroma()
#     qa_chain = build_qa_chain(vectorstore)

#     uploaded_file = st.file_uploader("Upload your contract (.txt)", type=["txt"])
#     if uploaded_file is not None:
#         contract_text = uploaded_file.read().decode("utf-8")

#         # Split into chunks
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         contract_chunks = splitter.split_text(contract_text)

#         st.subheader("🔍 Clause Analysis")
#         for i, chunk in enumerate(contract_chunks):
#             st.markdown(f"**Clause {i+1}:**")
#             st.write(chunk)

#             query = f"""
#             Analyze this clause for legal risks: {chunk}.
#             1. Identify if it's risky (yes/no).
#             2. If risky, explain why.
#             3. Suggest a safer alternative clause.
#             4. Suggest negotiation wording for a fair balance.
#             """
#             result = qa_chain.run(query)
#             st.success(result)


# if __name__ == "__main__":
#     main()
