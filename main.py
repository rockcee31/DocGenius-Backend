from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import uuid
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL(s) in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_filename, "wb") as f:
        contents = await file.read()
        f.write(contents)

    loader = PyPDFLoader(file_path=temp_filename)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    split_docs = text_splitter.split_documents(documents=docs)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="https://qdrant-vector-db.onrender.com",
        collection_name="learning_vectors",
        embedding=embedding_model
    )

    os.remove(temp_filename)

    if vector_store:
        return {"status": "uploaded"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_db = QdrantVectorStore.from_existing_collection(
        url="https://qdrant-vector-db.onrender.com",
        collection_name="learning_vectors",
        embedding=embedding_model
    )

    query = messages[-1].get('content')

    search_results = vector_db.similarity_search(query=query)

    context = "\n\n\n".join([
        f"PageContent: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
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

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return {"answer": chat_completion.choices[0].message.content}

@app.get('/')
async def root():
    return {"message": "Hello chai code"}

def main():
    print("Hello from backend!")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
