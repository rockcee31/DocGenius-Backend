from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import uuid
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai  # Updated
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize once — not inside routes
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
qdrant_client = QdrantClient(
    url="https://qdrant-vector-db.onrender.com:6333",
    timeout=30
)

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    user_id = uuid.uuid4().hex
    collection_name = f"learning_vectors_{user_id}"
    temp_filename = f"temp_{user_id}.pdf"
    
    try:
        # Save the file locally
        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        # Load and split PDF
        loader = PyPDFLoader(file_path=temp_filename)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        split_docs = text_splitter.split_documents(documents=docs)

        # Initialize vector store (do NOT return or store this object)
        QdrantVectorStore.from_documents(
            documents=split_docs,
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embedding_model
        )

        # ✅ Only return basic info
        return {"status": "uploaded", "collection_name": collection_name}

    except Exception as e:
        logging.exception("Upload failed")
        return {"status": "error", "message": f"Upload processing failed: {str(e)}"}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# Chat Endpoint
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages")
        collection_name = data.get("collection_name")  # must come from client!

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        if not collection_name:
            raise HTTPException(status_code=400, detail="collection_name is required")

        query = messages[-1].get('content')
        if not query:
            raise HTTPException(status_code=400, detail="Query content missing in last message")

        vector_db = QdrantVectorStore.from_existing_collection(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embedding_model
        )

        search_results = vector_db.similarity_search(query=query)

        context = "\n\n\n".join([
            f"PageContent: {result.page_content}\n"
            f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
            f"File Location: {result.metadata.get('source', 'Unknown')}"
            for result in search_results
        ])

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the available context retrieved from
        PDF files along with page content and page number.

        You should only answer the user based on the following context and guide the user to open the correct page number for more info.

        Context:
        {context}
        """

        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return {"answer": chat_completion.choices[0].message["content"]}

    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logging.error(f"Unexpected error in /chat: {e}", exc_info=True)
        return {"error": "An error occurred while processing your request."}

# Root Endpoint
@app.get('/')
async def root():
    return {"message": "Hello chai code"}

# App runner
def main():
    print("Hello from backend!")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
