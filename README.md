#  AI Document Assistant (RAG-based)

An AI-powered application that allows users to upload PDF documents and interact with them using natural language queries.

Built using Retrieval-Augmented Generation (RAG) with Groq LLM for fast and accurate responses.

---

## 🚀 Features

- 📂 Upload and process PDF documents
- 💬 Chat with your document
- ⚡ Fast responses using Groq LLM
- 🧠 Semantic search using vector embeddings
- 📊 Context-aware answers using RAG pipeline
- 💡 Clean chat-based UI with history

---

## 🧠 How it Works (RAG Pipeline)

1. PDF is loaded and split into chunks  
2. Each chunk is converted into embeddings  
3. Stored in FAISS vector database  
4. User query is embedded and matched  
5. Relevant chunks are retrieved  
6. LLM generates answer using retrieved context  

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS (Vector DB)
- HuggingFace Embeddings
- Groq API (LLM)

---

## 🎯 Use Cases

- Chat with PDFs
- Knowledge base assistant
- Document search system
- Research assistant

---

## 🚀 Future Improvements

- Multi-document support
- Conversation memory
- Deployment (Streamlit Cloud / Render)
- Authentication system

---

## 👨‍💻 Author

Sravana A J