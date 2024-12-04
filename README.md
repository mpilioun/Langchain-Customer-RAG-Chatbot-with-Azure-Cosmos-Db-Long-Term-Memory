# Lanchain RAG Chatbot with FastAPI Backend and Azure Cosmos DB Session Storage

This repository contains a simple **Retrieval-Augmented Generation (RAG) Customer Support Chatbot** built using FastAPI. The chatbot is designed to answer customer queries based on provided documents. It leverages **LangChain** for the RAG framework and uses **Azure Cosmos DB** for storing session information after every message exchange, ensuring persistent and context-aware interactions.

---

## Overview

The RAG chatbot implements the following workflow:

1. **Document Ingestion**: User-provided documents are processed and stored in a retriever-friendly format.
2. **Retrieval**: The chatbot retrieves relevant pieces of information from the ingested documents based on user questions.
3. **Augmentation**: The retrieved information is used to craft responses using natural language generation.
4. **Session Storage**: Each message exchange is logged in **Azure Cosmos DB** for persistent sessions, enabling continuity in multi-turn conversations.
5. **FastAPI Backend**: A robust API backend facilitates the chatbot's operation and serves as the integration layer for other applications.

---

## Key Features

### 🔍 **RAG (Retrieval-Augmented Generation) Framework**
- Combines information retrieval with generative AI to provide accurate and contextually relevant answers.
- Powered by **LangChain**, making it easy to scale and customize the retrieval and augmentation processes.

### ⚡ **FastAPI Backend**
- Lightweight, high-performance API service built with Python's FastAPI framework.
- Supports integration with web apps, mobile apps, and other systems via RESTful endpoints.

### ☁️ **Azure Cosmos DB Integration**
- Stores chat sessions, including the context of multi-turn conversations.
- Ensures scalability and high availability with Cosmos DB's global distribution capabilities.

### 📄 **Document-Based Q&A**
- Provides answers specifically tailored to the content of uploaded documents.
- Supports multiple document formats, enabling diverse use cases such as FAQ systems or document support.

### 🗂 **Session Persistence**
- Tracks and saves user interactions in a structured manner for seamless and context-aware conversations.

### 🚀 **Scalable and Extensible**
- Modular architecture allows easy integration of additional retrievers, language models, or document types.
- Ready for deployment in production environments with support for Docker and cloud platforms.

---

### Installation 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mpilioun/Langchain-Customer-RAG-Chatbot-with-Azure-Cosmos-Db-Long-Term-Memory.git
   cd rag-chatbot

2. **Set up virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env_sample` to `.env`
   - Add your configuration values

### Running the Application 🚀

1. **Start the Backend**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```


## Open to Suggestions
I am always open to comments and potential improvements to make this project better. Feel free to share your thoughts, ideas, or feedback. Let's collaborate to enhance the capabilities and usability of this chatbot!

## 🪪 License

This project is licensed under the MIT License. See the LICENSE file for more details.
