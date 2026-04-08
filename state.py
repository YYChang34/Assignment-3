from typing import TypedDict


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    grade: str