import streamlit as st
import os
import tempfile
import pymupdf as fitz
from rag_pipeline import create_vectorstore
from graph_pipeline import build_graph
from langchain_core.messages import HumanMessage, AIMessage

def extract_text_from_pdf_app(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    doc.close()
    return text

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(90deg,#1a1a2e,#16213e);
color:white;border-radius:10px;margin-bottom:20px;">
    <h1>🤖 RAG Assistant</h1>
    <p>Posez des questions sur vos documents PDF</p>
    <p style="font-size:12px;color:#aaa;">LangChain · ChromaDB · Groq · HuggingFace · LangGraph</p>
</div>
""", unsafe_allow_html=True)

# Initialisation session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Uploadez votre PDF",
        type=["pdf"],
        help="Formats supportés : PDF"
    )

    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            with st.spinner("Traitement du PDF en cours..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                text = extract_text_from_pdf_app(tmp_path)
                vectorstore = create_vectorstore(text)
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                docs = retriever.invoke("de quoi parle ce document")
                context_preview = "\n".join(
    doc.page_content[:300] for doc in docs
)
                graph = build_graph(retriever, context_preview)
                st.session_state.rag_chain = graph
                st.session_state.chat_history = []
                st.session_state.pdf_processed = True
                os.unlink(tmp_path)

            st.success("✅ PDF traité avec succès !")

    if st.session_state.pdf_processed:
        st.info("📄 Document chargé")
        if st.button("🗑️ Nouveau document"):
            st.session_state.chat_history = []
            st.session_state.rag_chain = None
            st.session_state.pdf_processed = False
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("""
    **Comment utiliser :**
    1. Uploadez un PDF
    2. Attendez le traitement
    3. Posez vos questions !
    """)
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#666;font-size:12px;">
    Développé par <b>Omer Bokassa Boueke</b><br>
    ML Engineer & MLOps
    </div>
    """, unsafe_allow_html=True)

# Zone chat
if not st.session_state.pdf_processed:
    st.info("👈 Uploadez un PDF dans la barre latérale pour commencer")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📋 Contrats**\nAnalysez vos contrats")
    with col2:
        st.markdown("**📊 Rapports**\nInterrogez vos rapports")
    with col3:
        st.markdown("**📚 Documents**\nPosez des questions")

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Posez votre question..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                result = st.session_state.rag_chain.invoke({
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                    "context": "",
                    "answer": "",
                    "is_relevant": False
                })
                response = result["answer"]
                sources = result.get("source_documents", [])
                st.markdown(response)

                if sources:
                    with st.expander("📚 Sources utilisées"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1} :**")
                            st.text(doc.page_content[:200] + "...")
                            st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response)
        ])