# RAG Assistant : LangChain + LangGraph + Groq

> Assistant IA intelligent sur vos documents PDF, avec détection automatique des questions hors sujet via LangGraph.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-red)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-✓-blue)](https://docker.com)

---;                                   

## Problématique

Les LLM comme GPT ou Llama ont un problème fondamental : ils ne connaissent pas vos documents internes et peuvent halluciner. Ce projet résout ce problème avec RAG (Retrieval Augmented Generation) : le modèle répond en se basant uniquement sur vos documents, sans inventer.

---

## Démo en ligne

**[rag-assistant-dkc1.onrender.com](https://rag-assistant-dkc1.onrender.com)**

> Le free tier Render peut mettre 50 secondes à démarrer après inactivité.

---

## Architecture

```
PDF uploadé
     ↓
PyMuPDF : extraction du texte
     ↓
LangChain : découpage en chunks (500 tokens, overlap 50)
     ↓
HuggingFace : transformation en embeddings
     ↓
ChromaDB : stockage des vecteurs
     ↓
Question utilisateur
     ↓
LangGraph : analyse si la question est pertinente
     ↓
OUI → ChromaDB : recherche sémantique (top 3 chunks)
    → Groq (LLM) : génération de la réponse
NON → Réponse polie "hors sujet"
```

---

## LangGraph : Intelligence du flux

Sans LangGraph, toute question reçoit une réponse même si elle n'a rien à voir avec le document. LangGraph ajoute un noeud d'analyse qui décide du bon chemin :

```
analyze_question → OUI → retrieve_and_answer → END
                 → NON → handle_off_topic    → END
```

---

##  Stack technique

| Composant | Rôle |
|-----------|------|
| PyMuPDF | Extraction du texte PDF |
| LangChain | Orchestration du pipeline RAG |
| LangGraph | Gestion intelligente du flux |
| HuggingFace | Modèle d'embeddings (all-MiniLM-L6-v2) |
| ChromaDB | Base vectorielle |
|Groq | LLM (openai/gpt-oss-120b)|
| Streamlit | Interface utilisateur |
| Docker | Conteneurisation |
| Render | Déploiement cloud |

---

## 📁 Structure du projet

```
rag-assistant/
├── app.py              ← Interface Streamlit
├── rag_pipeline.py     ← Extraction PDF + ChromaDB
├── graph_pipeline.py   ← LangGraph (3 noeuds)
├── data/
│   ├── uploads/        ← PDFs temporaires
│   └── chroma_db/      ← Base vectorielle
├── Dockerfile
├── requirements.txt
├── .env                ← GROQ_API_KEY (non versionné)
└── .gitignore
```

---

## Lancer en local

```bash
# Cloner le repo
git clone https://github.com/Bokassa09/rag-assistant.git
cd rag-assistant

# Créer l'environnement
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Configurer la clé API
echo "GROQ_API_KEY=ta-clé-groq" > .env

# Lancer
streamlit run app.py
```

### Avec Docker

```bash
docker build -t rag-assistant .
docker run -p 8501:8501 -e GROQ_API_KEY=ta-clé rag-assistant
```

---

### Évaluation du système RAG

Une première évaluation a été réalisée à l’aide du script `evaluate_rag.py`, qui pose 3 questions à partir d’un PDF de test. Un LLM (`openai/gpt-oss-120b`) évalue ensuite chaque réponse sur deux métriques, avec un score compris entre 0 et 1.

| Métrique         | Score moyen |
| ---------------- | ----------: |
| Faithfulness     |    **0,67** |
| Answer Relevancy |    **0,95** |

Ces premiers résultats montrent une **très bonne pertinence des réponses**, avec un score élevé en Answer Relevancy. En revanche, le score de Faithfulness indique qu’il reste une marge d’amélioration concernant l’alignement des réponses avec le contexte fourni.

**Limites :** cette évaluation repose seulement sur 3 questions et ne mesure pas la qualité de la récupération des documents. Les métriques **Context Precision** et **Context Recall** ne sont notamment pas encore évaluées.

**Prochaine étape :** mettre en place une évaluation plus complète avec **RAGAS**, comprenant davantage de questions et un protocole d’évaluation plus strict, afin de mesurer à la fois la qualité de la récupération du contexte et celle des réponses générées.

---

## 👤 Auteur

**Omer Bokassa Boueke** : ML Engineer & MLOps

- 🔗 [GitHub](https://github.com/Bokassa09)
- 🔗 [LinkedIn](https://linkedin.com/in/omer-bokassa-boueke-9674a331a)
- 🔗 [Customer Churn Platform](https://customer-churn-platform-1-zeg0.onrender.com)
