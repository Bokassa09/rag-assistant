import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

# 1. ÉTAT DU GRAPHE
class RAGState(TypedDict):
    question: str
    chat_history: List
    context: str
    answer: str
    is_relevant: bool

# 2. NOEUDS

def analyze_question(state: RAGState) -> RAGState:
    """Noeud 1 : la question est-elle pertinente ?"""
    question = state["question"]
    context_preview = state.get("context", "")[:300]

    prompt = f"""Tu analyses si une question concerne 
un document personnel ou professionnel.

Aperçu du document : {context_preview}

Question : {question}

Sois TRÈS LARGE dans ton interprétation.
Une question sur une personne, ses compétences,
son parcours, son nom est TOUJOURS pertinente
pour un document de présentation.

Réponds UNIQUEMENT par OUI ou NON."""


    response = llm.invoke(prompt)
    is_relevant = "OUI" in response.content.upper()

    return {**state, "is_relevant": is_relevant}

def retrieve_and_answer(state: RAGState, retriever) -> RAGState:
    """Noeud 2 : chercher et répondre"""
    question = state["question"]
    chat_history = state["chat_history"]

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = ""
    for msg in chat_history[-4:]:
        if isinstance(msg, HumanMessage):
            history_text += f"Humain : {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant : {msg.content}\n"

    prompt = f"""Tu es un assistant utile.
Réponds en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Historique :
{history_text}

Contexte :
{context}

Question : {question}

Réponse :"""

    response = llm.invoke(prompt)

    return {
        **state,
        "context": context,
        "answer": response.content,
        "source_documents": docs
    }

def handle_off_topic(state: RAGState) -> RAGState:
    """Noeud 3 : question hors sujet"""
    return {
        **state,
        "answer": "Je suis conçu pour répondre uniquement "
                  "aux questions sur le document chargé. "
                  "Votre question ne semble pas en rapport "
                  "avec ce document. Pouvez-vous reformuler ?"
    }

def should_retrieve(state: RAGState) -> str:
    """Décision : quelle branche ?"""
    return "retrieve" if state["is_relevant"] else "off_topic"

# 3. CONSTRUIRE LE GRAPHE
def build_graph(retriever, context_preview: str = ""):

    def retrieve_node(state):
        return retrieve_and_answer(state, retriever)

    def analyze_node(state):
        state["context"] = context_preview
        return analyze_question(state)

    graph = StateGraph(RAGState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("off_topic", handle_off_topic)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "off_topic": "off_topic"
        }
    )

    graph.add_edge("retrieve", END)
    graph.add_edge("off_topic", END)

    return graph.compile()
