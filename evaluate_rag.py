import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag_pipeline import create_vectorstore, extract_text_from_pdf
from graph_pipeline import build_graph

load_dotenv()

import re

def parse_score(response_text: str) -> float:
    matches = re.findall(r'\d+\.?\d*', response_text.strip())
    if matches:
        score = float(matches[0])
        if score > 1:
            score = score / 10
        return min(max(score, 0), 1)
    return 0.5

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def evaluate_faithfulness(question, answer, context):
    prompt = f"""Évalue si la réponse est basée UNIQUEMENT 
sur le contexte fourni. Réponds avec un nombre entre 0 et 1.

Contexte : {context}
Question : {question}
Réponse : {answer}

Score (0 = inventé, 1 = basé sur le contexte) :"""
    
    response = llm.invoke(prompt)
    return parse_score(response.content)

def evaluate_relevancy(question, answer):
    prompt = f"""Évalue si la réponse répond directement 
à la question. Réponds avec un nombre entre 0 et 1.

Question : {question}
Réponse : {answer}

Score (0 = hors sujet, 1 = très pertinent) :"""
    
    response = llm.invoke(prompt)
    return parse_score(response.content)

def evaluate_rag(pdf_path: str, eval_data: list):
    """Évaluation complète du RAG"""
    
    print("Chargement du PDF...")
    text = extract_text_from_pdf(pdf_path)
    vectorstore = create_vectorstore(text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs_preview = retriever.invoke("de quoi parle ce document")
    context_preview = "\n".join(
        doc.page_content[:300] for doc in docs_preview
    )
    graph = build_graph(retriever, context_preview)
    
    results = []
    
    for item in eval_data:
        question = item["question"]
        print(f"\nQuestion : {question}")
        
        result = graph.invoke({
            "question": question,
            "chat_history": [],
            "context": "",
            "answer": "",
            "is_relevant": False
        })
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        context = "\n".join(
            doc.page_content for doc in source_docs
        )
        
        faithfulness = evaluate_faithfulness(
            question, answer, context
        )
        relevancy = evaluate_relevancy(question, answer)
        
        results.append({
            "question": question,
            "answer": answer[:100] + "...",
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy
        })
        
        print(f"Faithfulness    : {faithfulness:.2f}")
        print(f"Answer Relevancy: {relevancy:.2f}")
    
    print("\n--- Résultats finaux ---")
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relev = sum(r["answer_relevancy"] for r in results) / len(results)
    print(f"Faithfulness moyen     : {avg_faith:.2f}")
    print(f"Answer Relevancy moyen : {avg_relev:.2f}")
    
    return results

if __name__ == "__main__":
    eval_data = [
        {"question": "De quoi parle ce document ?"},
        {"question": "Quelle est l'expérience professionnelle ?"},
        {"question": "Quelles sont les compétences techniques ?"}
    ]
    
    evaluate_rag("data/uploads/test.pdf", eval_data)
