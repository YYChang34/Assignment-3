import os
from langgraph.graph import END, StateGraph
from langchain_chroma import Chroma

from config import get_embeddings, DB_FOLDER, FILES, EMBEDDING_MODEL_2
from state import AgentState
from task_a import run_legacy_agent as _run_legacy_agent
from task_b import build_retrieve_node
from task_c import grade_documents_node
from task_d import rewrite_node
from task_e import generate_node


def _get_active_db_folder(model_name: str, chunk_size: int = 2000) -> str:
    model_short = model_name.split("/")[-1]
    return os.path.join(DB_FOLDER, f"{model_short}_chunk{chunk_size}")


def initialize_vector_dbs(model_name: str = EMBEDDING_MODEL_2, chunk_size: int = 2000) -> dict:
    embeddings = get_embeddings(model_name)
    db_root = _get_active_db_folder(model_name, chunk_size)
    retrievers = {}

    for key in FILES.keys():
        persist_dir = os.path.join(db_root, key)
        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 8})
        else:
            print(f"Error: Database for '{key}' not found at {persist_dir}")

    return retrievers


RETRIEVERS = initialize_vector_dbs(chunk_size=int(os.getenv("CHUNK_SIZE", "2000")))


def build_graph():
    retrieve_node = build_retrieve_node(RETRIEVERS)

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state: AgentState) -> str:
        if state["grade"] == "yes":
            return "generate"
        if state["search_count"] > 2:
            print("Max retries reached")
            return "generate"
        return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_graph_agent(question: str) -> str:
    app = build_graph()
    inputs = {
        "question": question,
        "search_count": 0,
        "grade": "no",
        "documents": "",
        "generation": "",
    }
    result = app.invoke(inputs)
    return result["generation"]


def run_legacy_agent(question: str) -> str:
    return _run_legacy_agent(question, RETRIEVERS)
