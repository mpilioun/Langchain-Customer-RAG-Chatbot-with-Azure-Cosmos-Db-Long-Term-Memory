import datetime
import os
from dotenv import load_dotenv
from typing import cast
from fastapi import Body, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError

load_dotenv()

# Initialize app
app = FastAPI()

# ######################################### Comsos DB Connections #############################

# Initialize the Azure Cosmos DB client
cosmos_client = CosmosClient(
    os.getenv('AZURE_COSMOSDB_ENDPOINT'), 
    credential=os.getenv('AZURE_COSMOSDB_KEY')
)

# Initialize the Azure Cosmos DB database client
database = cosmos_client.get_database_client(os.getenv('AZURE_COSMOSDB_DATABASE_NAME'))

# Initialize the Azure Cosmos DB container client
container = database.get_container_client(os.getenv('AZURE_COSMOSDB_CONTAINER_NAME'))
active_container = database.get_container_client(os.getenv('AZURE_COSMOSDB_CONTAINER_NAME_ACTIVE'))

# ############################################## RAG LLM ######################################

from chatbot.rag_chain import call_rag_model,RagState



# ################################### Class Models ############################################
class Input(BaseModel):
    question: str

class Metadata(BaseModel):
    session_id: str
    customer_id: str

class Config(BaseModel):
    metadata: Metadata
    
class RequestBody(BaseModel):
    input: Input 
    config: Config

# ######################################   APP EndPoints   ####################################

@app.get("/", description="Initial Prompt")
async def root():
    return {"message": "Hello! I am Sophia, your helpful assistant. I can assist with any information you need regarding the company. How can I help you today?"}



@app.post("/chatbot")
async def chat(
    query: RequestBody = Body(...),
):

    session_id = query.config.metadata.session_id
    customer_id = query.config.metadata.customer_id

    # Try to fetch all active sessions for the customer
    try:
        active_sessions = active_container.query_items(
            query=f"SELECT * FROM c WHERE c.customer_id = @customer_id",
            parameters=[{"name": "@customer_id", "value": customer_id}],
            enable_cross_partition_query=True
        )

        # Look for an existing session that is not the same as the current session_id
        old_sessions = []
        for item in active_sessions:
            if item["session_id"] != session_id:
                old_sessions.append(item)
                
        if session_id not in old_sessions and len(old_sessions) >=1:
            for old_session in old_sessions:
                await archive_interaction(
                    session_id=old_session["session_id"],
                    customer_id=customer_id
                )
                await delete_interaction(
                    session_id=old_session["session_id"],
                    customer_id=customer_id
                )

        try:
            existing_item = active_container.read_item(item=session_id, partition_key=customer_id)
            state = cast(RagState, existing_item)
            state["context"] = {}
            state["input"] = query.input.question
        except Exception:
            # If no record exists for the session_id, create a new state manually
            state = {
                "session_id": session_id,
                "customer_id": customer_id,
                "input": query.input.question,
                "chat_history": [],
                "context": {},
                "answer": ""
            }

    except Exception as e:
        # If no active sessions exist for the customer, create a new state
        state = {
            "session_id": session_id,
            "customer_id": customer_id,
            "input": query.input.question,
            "chat_history": [],
            "context": {},
            "answer": ""
        }

    response = await call_rag_model(state)

    stripped_state = state
    stripped_state.pop("context", None)

    interaction_data = {
        'id': state["session_id"], 
        'session_id': state["session_id"], 
        'chat_history': [
            {
                'role': msg["role"], 
                'content': msg["content"],
                'timestamp': msg["timestamp"]
            } for msg in state["chat_history"]
        ],
        'customer_id': state["customer_id"],
    }
    active_container.upsert_item(interaction_data)

    return response["answer"]



@app.post("/archive_interaction")
async def archive_interaction(session_id: str,customer_id: str):
    try:
        state = active_container.read_item(item=session_id, partition_key=customer_id)
        current_time = datetime.datetime.now(datetime.timezone.utc)
        interaction_data = {
            'id': state["session_id"], 
            'session_id': state["session_id"],
            'chat_history': [
                {
                    'sender': msg["role"], 
                    'content': msg["content"],
                    'timestamp': msg["timestamp"]
                } for msg in state["chat_history"]
            ],
            'end_timestamp': current_time.isoformat(),
            'customer_id': state["customer_id"]
        }
        
        container.upsert_item(interaction_data)
        return {"status": "success", "message": "Interaction data saved successfully."}
    # Handle the specific case where the item is not found
    except CosmosResourceNotFoundError:
        return {
            "status": "not_found",
            "message": (
                f"Entity with session_id '{session_id}' does not exist for customer_id '{customer_id}'. "
                "This may be because the session was only initialized and with no messages ever sent by the user, "
                "resulting in only a placeholder session being created."
            )
        }  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving interaction data: {str(e)}")

@app.post("/delete_active_interaction")
async def delete_interaction(session_id: str,customer_id: str):
    try:
        active_container.delete_item(item=session_id, partition_key=customer_id)

        return {"status": "success", "message": "Session state deleted successfully from active container."}
    
    # Handle the specific case where the item is not found
    except CosmosResourceNotFoundError:
        return {
            "status": "not_found",
            "message": (
                f"Entity with session_id '{session_id}' does not exist for customer_id '{customer_id}'. "
                "This may be because the session was only initialized and with no messages ever sent by the user, "
                "resulting in only a placeholder session being created."
            )
        }   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session state data: {str(e)}")