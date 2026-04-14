"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Streamlit Interactive UI
"""

import streamlit as st
import requests
import json

# Page Config
st.set_page_config(
    page_title="DeepChain Hybrid RAG",
    page_icon="🕸️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ System Settings")
rag_method = st.sidebar.selectbox("Retrieval Method", ["GraphRAG (Hybrid)", "Naive RAG (Vector)"])
top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 5)
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

if st.sidebar.button("🚀 Trigger Ingestion"):
    with st.spinner("Processing documents..."):
        try:
            res = requests.post(f"{api_url}/ingest")
            st.sidebar.success(f"Ingested {res.json()['entities_extracted']} entities!")
        except Exception as e:
            st.sidebar.error("Ingestion failed.")

# Main UI
st.markdown('<h1 class="main-header">DeepChain Hybrid RAG Explorer</h1>', unsafe_allow_html=True)
st.write("Extracting intelligence from financial data using Knowledge Graphs and Vector Embeddings.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query Input
if prompt := st.chat_input("Ask a financial question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing graph and vector records..."):
            try:
                method_code = "graph" if "Graph" in rag_method else "naive"
                response = requests.post(
                    f"{api_url}/query",
                    json={"question": prompt, "method": method_code, "top_k": top_k}
                )
                answer = response.json()["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info(
    "DeepChain-Hybrid-RAG uses Neo4j for structural knowledge and Weaviate for semantic search. "
    "Designed for complex multi-hop financial queries."
)
