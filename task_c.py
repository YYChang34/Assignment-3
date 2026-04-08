from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm, retry_logic
from state import AgentState


@retry_logic
def grade_documents_node(state: AgentState) -> dict:
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a strict relevance grader for a financial RAG system.
Your task: determine if the retrieved document context contains information useful for answering the user's question.

A document is RELEVANT (output "yes") if it:
- Contains specific financial figures, metrics, or tables related to the question
- Mentions the company and time period (e.g., fiscal year 2024) asked about
- Includes factual content that directly helps answer the question

A document is IRRELEVANT (output "no") if it:
- Discusses a completely different company or unrelated topic
- Only covers a different time period (e.g., only 2022/2023 data when 2024 is asked)
- Contains no useful financial information for the question

CRITICAL: Output ONLY one word — either "yes" or "no". No explanation allowed."""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved document context:\n\n{documents}\n\nUser question: {question}"),
    ]

    response = llm.invoke(msg)
    content = response.content.strip().lower()
    grade = "yes" if "yes" in content else "no"
    print(f"   Relevance Grade: {grade}")
    return {"grade": grade}
