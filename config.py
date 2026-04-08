import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv(override=True)

retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
FILES = {
    "apple": "FY24_Q4_Consolidated_Financial_Statements.pdf",
    "tesla": "tsla-20241231-gen.pdf"
}

EMBEDDING_MODEL_1 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_2 = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings(model_name: str = EMBEDDING_MODEL_2) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

def get_llm(temperature: float = 0):
    api_key = os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=api_key
    )
