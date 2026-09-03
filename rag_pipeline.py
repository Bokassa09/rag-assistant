import os
from dotenv import load_dotenv
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extraire le texte d'un PDF avec PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    doc.close()
    return text

def create_vectorstore(text: str) -> Chroma:
    """Chunking + Embeddings + ChromaDB"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    print(f"Nombre de chunks créés : {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )
    return vectorstore

def create_rag_chain(vectorstore: Chroma):
    """Créer la chaîne RAG avec LangChain 1.3+"""

    llm = ChatGroq(
    model="groq/compound-mini",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant utile qui répond aux questions 
en te basant UNIQUEMENT sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte,
dis-le clairement sans inventer.

Contexte : {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(input_dict):
        question = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        docs = retriever.invoke(question)
        context = format_docs(docs)
        messages = prompt.format_messages(
            context=context,
            chat_history=chat_history,
            input=question
        )
        response = llm.invoke(messages)
        return {"answer": response.content, "source_documents": docs}

    return get_response

def process_pdf(pdf_path: str):
    """Pipeline complet : PDF → RAG Chain"""
    print(f"Traitement du PDF : {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Texte extrait : {len(text)} caractères")
    vectorstore = create_vectorstore(text)
    chain = create_rag_chain(vectorstore)
    print("Pipeline RAG prêt !")
    return chain, [] 

def create_graph(vectorstore: Chroma):
    """Créer le graphe LangGraph"""
    from graph_pipeline import build_graph
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Aperçu du contenu pour analyze_question
    docs = retriever.invoke("de quoi parle ce document")
    context_preview = "\n".join(
        doc.page_content[:100] for doc in docs
    )
    
    graph = build_graph(retriever, context_preview)
    return graph