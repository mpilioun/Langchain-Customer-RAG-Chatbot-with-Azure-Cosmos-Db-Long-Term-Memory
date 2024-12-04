import os
import datetime
from dotenv import load_dotenv
from typing import Any, AsyncIterable, Dict, Sequence
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()



# Rag State
class RagState(TypedDict):
    session_id: str
    customer_id: str
    input: str
    chat_history: Annotated[Sequence[Dict[str, Any]], add_messages]
    context: str
    answer: str


# Rag Function

async def call_rag_model(state: RagState) -> AsyncIterable[str]:
    INDEX_NAME = os.environ.get("INDEX_NAME")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    embeddings_api_version = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION")
    embeddings_deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
    embeddings_api_key = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_KEY")
    embeddings_endpoint = os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment_name,
        openai_api_key=api_key,
        openai_api_version=api_version,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        model="gpt-4o-mini",
    )
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=embeddings_endpoint,
        azure_deployment=embeddings_deployment_name,
        openai_api_key=embeddings_api_key,
        openai_api_version=embeddings_api_version,
    )
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = docsearch.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks, and you should respond based on the provided retrieved context. "
        "If you cannot find an answer within this context, respond with a helpful message like: "
        "'I don't have the exact answer you're looking for, but I'm here to help with anything else I can related to the company! "
        "Feel free to open a ticket in Help Center if you need more assistance.' "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # If the chat history is empty, set the initial prompt message
    if not state["chat_history"]:
        answer = "Hello! I am Sophia, your helpful assistant. I can assist with any information you need regarding the company. How can I help you today?"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state["chat_history"] = state["chat_history"] + [{"role": "assistant", "content": answer, "timestamp": timestamp}]

    # Process the user input
    user_input_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    response = rag_chain.invoke(state)
    answer_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Update chat history with the correct structure
    updated_chat_history = state["chat_history"] + [
        {"role": "user", "content": state["input"], "timestamp": user_input_timestamp},
        {"role": "assistant", "content": response["answer"], "timestamp": answer_timestamp},
    ]

    state["chat_history"] = updated_chat_history
    state["context"] = response["context"]
    state["answer"] = response["answer"]
    
    return state
