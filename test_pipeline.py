from rag_pipeline import process_pdf

# Test du pipeline
pdf_path = "data/uploads/test.pdf"
chain, history = process_pdf(pdf_path)

# Test d'une question
question = "Omer dans le text ça a rapport ça quoi ?"
result = chain({
    "input": question,
    "chat_history": history
})

print(f"\nQuestion : {question}")
print(f"Réponse : {result['answer']}")