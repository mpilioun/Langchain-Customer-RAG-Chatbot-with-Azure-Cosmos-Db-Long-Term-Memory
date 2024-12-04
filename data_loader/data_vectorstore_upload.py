import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

INDEX_NAME = os.getenv('INDEX_NAME')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    
    print(f"Loaded a document with {len(pages)} pages")

    PineconeVectorStore.from_documents(
        pages, embeddings, index_name=INDEX_NAME
    )

    print("**** Loading document to Vectorstore Completed ***")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory
    documents_dir = os.path.join(current_dir, 'documents') # Construct the path to the 'documents' folder

    # Iterate over the files in the 'documents' folder
    for filename in os.listdir(documents_dir):
        doc_path = os.path.join(documents_dir, filename)    # Get the full path of each file
        ingest_docs(doc_path)
    
    print("Finished loading documents")
