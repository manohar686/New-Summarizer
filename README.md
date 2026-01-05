# 🔍 Retrieval-Augmented Question Answering (RAG) System using LangChain & Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that answers user questions using relevant document context instead of relying on the LLM’s internal knowledge, ensuring accurate and grounded responses.

The system integrates **LangChain**, **Ollama (LLaMA 3.2)**, **HuggingFace embeddings**, and **ChromaDB**, and uses a **map-reduce strategy** to safely handle long documents while preventing hallucinations.

---

## 📌 Project Overview

- Semantic search using vector embeddings  
- Retrieval-first answer generation  
- Map-and-Reduce strategy for long context handling  
- Local LLM inference using Ollama  
- Persistent vector storage with ChromaDB  

---

## 🏗️ Architecture Flow


---

## 🧠 File Descriptions

### ask_llm.py
- Entry point of the application  
- Initializes Ollama LLM (`llama3.2:1b`)  
- Loads vector store and prompts  
- Executes map-reduce retrieval logic  
- Returns final grounded response  

---

### loader_and_splitter.py
- Loads raw text documents  
- Splits content into small semantic chunks  
- Uses recursive splitting for better embedding quality  

**Key Settings**
- Chunk size: 100 characters  
- No chunk overlap  

---

### prompt_templates.py
Defines two strict prompts to prevent hallucinations:

**Map Prompt**
- Checks relevance of each chunk  
- Condenses only useful information  
- Returns `-` if no relevant data found  

**Reduce Prompt**
- Clearly answers the question using condensed chunks  
- Explicitly forbids making up information  

---

### retreival.py
Implements token-aware **Map & Reduce** logic:

- Retrieves top-k similar chunks  
- If combined text < token limit → direct answer  
- Else → map each chunk → reduce into final response  

This ensures:
- No token overflow  
- Better answer quality  
- Safe handling of large documents  

---

### vector_store.py
- Generates embeddings using HuggingFace MiniLM  
- Stores vectors in ChromaDB  
- Enables fast semantic similarity search  
- Persists embeddings locally for reuse  

**Embedding Model**

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **LLM:** Ollama (LLaMA 3.2)  
- **Framework:** LangChain  
- **Embeddings:** HuggingFace Transformers  
- **Vector Database:** ChromaDB  
- **Retrieval Method:** Semantic Search + Map-Reduce RAG  

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install langchain langchain-community langchain-core
pip install langchain-chroma langchain-huggingface
pip install chromadb sentence-transformers
pip install ollama
ollama run llama3.2:1b
python ask_llm.py
