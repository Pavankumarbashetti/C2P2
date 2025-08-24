import os
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    st.set_page_config(
        page_title="AI Contract Risk Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # -------------------------------
    # CUSTOM CSS
    # -------------------------------
    st.markdown("""
        <style>
        .stApp {background-color: #f9f9f9;}
        .header {font-size:2.5rem; font-weight:bold; color:#1f4e79; margin-bottom:10px;}
        .risk-box {padding:20px; border-radius:10px; color:white; font-weight:bold; text-align:center; width:250px;}
        .risk-high {background-color:red;}
        .risk-medium {background-color:orange;}
        .risk-low {background-color:green;}
        .analysis-box {background-color:#ffffff; padding:20px; border-radius:10px; box-shadow:0px 0px 10px rgba(0,0,0,0.1);}
        .button-box {margin-top:10px; margin-bottom:10px;}
        </style>
    """, unsafe_allow_html=True)

    # -------------------------------
    # HEADER
    # -------------------------------
    st.markdown('<div class="header">📜 AI Contract Risk Analyzer & Negotiator</div>', unsafe_allow_html=True)
    st.markdown("Upload a contract, and get a professional **summary**, risk assessment, and negotiation suggestions.")

    # -------------------------------
    # LOAD VECTORSTORE
    # -------------------------------
    vectorstore = load_chroma()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # -------------------------------
    # FILE UPLOADER
    # -------------------------------
    uploaded_file = st.file_uploader("📂 Upload your contract (.txt)", type=["txt"])
    if uploaded_file is not None:
        contract_text = uploaded_file.read().decode("utf-8")

        # Retrieve relevant CUAD risks
        similar_docs = retriever.get_relevant_documents(contract_text)
        cuad_context = "\n\n".join([doc.page_content for doc in similar_docs])

        # -------------------------------
        # PROMPT LLM
        # -------------------------------
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a contract risk analyzer. Use the CUAD dataset context below to ground your analysis. "
             "Respond in a professional note style, summarize the entire contract as a note."),
            ("human",
             "CUAD Risks Context:\n{cuad_context}\n\n"
             "Now analyze this contract:\n{contract_text}\n\n"
             "Provide a single summary note with these sections:\n"
             "Contract Summary:\n"
             "Risk Assessment: (Label as High Risk / Medium Risk / Low Risk)\n"
             "Key Risk Factors:\n"
             "Safer Alternatives:\n"
             "Negotiation Suggestions:\n")
        ])

        query = prompt_template.format_messages(
            cuad_context=cuad_context,
            contract_text=contract_text
        )

        try:
            result = llm.invoke(query)

            # -------------------------------
            # DETERMINE RISK LEVEL
            # -------------------------------
            risk_label = "LOW"
            if "High Risk" in result.content:
                risk_label = "HIGH"
                risk_class = "risk-box risk-high"
            elif "Medium Risk" in result.content:
                risk_label = "MEDIUM"
                risk_class = "risk-box risk-medium"
            else:
                risk_class = "risk-box risk-low"

            # -------------------------------
            # TOP-RIGHT RISK BOX
            # -------------------------------
            st.markdown(f"""
            <div style="position: fixed; top: 100px; right: 20px;">
                <div class='{risk_class}'>⚠️ Contract Risk Level: {risk_label}</div>
            </div>
            """, unsafe_allow_html=True)
            # -------------------------------
            # RELEVANT CUAD CHUNKS BUTTON
            # -------------------------------
            # -------------------------------
            # CREATE TWO COLUMNS
            # -------------------------------
            col1, col2 = st.columns([2, 1])  # Summary gets more space

            # -------------------------------
            # RELEVANT CUAD CHUNKS BUTTON
            # -------------------------------
            show_cuad = col2.button("📚 Show Matched Source Data (CUAD Chunks)")

            # -------------------------------
            # CONTRACT SUMMARY
            # -------------------------------
            with col1:
                st.markdown("### ✅ Contract Analysis Summary")
            st.markdown(f"<div class='analysis-box'>{result.content}</div>", unsafe_allow_html=True)

            # -------------------------------
            # SHOW CUAD MATCHES IN SECOND COLUMN
            # -------------------------------
            if show_cuad:
                    st.markdown("### Relevant CUAD Matches")
                    for j, doc in enumerate(similar_docs):
                        st.markdown(f"**Match {j+1}:** {doc.page_content}")
            # if st.button("📚 Show Matched Source Data (CUAD Chunks)"):
            #     st.markdown("### Relevant CUAD Matches")
            #     for j, doc in enumerate(similar_docs):
            #         st.markdown(f"**Match {j+1}:** {doc.page_content}")
            # # -------------------------------
            # # CONTRACT SUMMARY
            # # -------------------------------
            # st.markdown("### ✅ Contract Analysis Summary")
            # st.markdown(f"<div class='analysis-box'>{result.content}</div>", unsafe_allow_html=True)

            

        except Exception as e:
            st.error(f"Error analyzing contract: {e}")

if __name__ == "__main__":
    main()
