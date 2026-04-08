from langchain_core.messages import HumanMessage

from config import get_llm, retry_logic
from state import AgentState


@retry_logic
def rewrite_node(state: AgentState) -> dict:
    question = state["question"]
    llm = get_llm()

    msg = [
        HumanMessage(content=(
            f"The search query '{question}' returned irrelevant results from a financial document database.\n\n"
            f"Rewrite this query using precise SEC 10-K financial terminology to improve retrieval accuracy.\n\n"
            f"CRITICAL RULES:\n"
            f"1. ALWAYS preserve the company name (Apple or Tesla) in the rewritten query.\n"
            f"2. Add 'fiscal year 2024' if a year is not already specified.\n"
            f"3. Use official SEC 10-K terminology.\n\n"
            f"Examples of transformations:\n"
            f"- 'Tesla capital expenditures' -> 'Tesla Capital Expenditures purchases of property plant and equipment fiscal year 2024'\n"
            f"- 'Apple spending on new tech' -> 'Apple Research and Development (R&D) expenses fiscal year 2024'\n"
            f"- 'Tesla revenue' -> 'Tesla Total revenues Net sales fiscal year 2024'\n"
            f"- 'Apple profit' -> 'Apple Net income fiscal year 2024'\n\n"
            f"Output ONLY the rewritten question text, nothing else."
        ))
    ]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}
