import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Cinema Analyst API")

# Insert your generated Ollama API key here
os.environ["OLLAMA_API_KEY"] = "23f7a840621b4b7da671e805f045899a"

# Initialize Ollama Cloud LLM 
llm = ChatOpenAI(
    model="gemma4:31b",
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url="https://ollama.com/v1",
    temperature=0
)

# --- BULLETPROOF DATA GENERATION ---
# This automatically creates the folder and file so it never fails!
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True) 

file_path = os.path.join(data_dir, "telugu_box_office.txt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write("The Paradise: Nani's highest-grossing film to date, surpassing 136 crore worldwide in September 2026.\n")
    f.write("Vadhala: The psychological thriller released digitally on Amazon Prime Video starting August 14, 2026.\n")
    f.write("Irumudi: Debuted on Netflix on September 18, 2026.\n")
    f.write("Sankranthiki Vasthunam: The highest-grossing 2025 movie in Telugu states with over 300 crore worldwide.\n")

# Load Telugu cinema data
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load_and_split()
# -----------------------------------

# Create fast local vector index
embeddings = FakeEmbeddings(size=1536)
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Setup prompt & QA chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based strictly on the provided context:
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_cinema_question(req: QueryRequest):
    response = qa_chain.invoke({"input": req.question})
    return {"answer": response["answer"]}