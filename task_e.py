from langchain_core.prompts import ChatPromptTemplate

from config import get_llm, retry_logic
from state import AgentState


@retry_logic
def generate_node(state: AgentState) -> dict:
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    if not documents.strip():
        return {"generation": "I don't know based on the provided documents. The query did not match any available financial data."}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert financial analyst. Your job is to answer questions based ONLY on the provided context.\n\n"
            "RULES:\n"
            "1. ALWAYS cite the source using the exact format: [Source: Apple 10-K] or [Source: Tesla 10-K].\n"
            "2. If the context does not contain the specific information needed, respond: "
            "\"I don't know based on the provided documents.\"\n"
            "3. Do NOT hallucinate, guess, or infer financial figures not explicitly stated in the context.\n"
            "4. When answering comparison questions, address both companies explicitly.\n"
            "5. Report numbers precisely as they appear in the documents (e.g., $391,035 million or $391 billion).\n"
            "6. Your answer must be in English.\n"
            "7. For percentage figures (e.g. gross margin %), prefer values explicitly stated in the context. "
            "If you must calculate a percentage, show the formula clearly as: "
            "(numerator ÷ denominator × 100%) and double-check the arithmetic before reporting.\n"
            "8. For capital expenditures or cash flow items, look for terms such as "
            "'purchases of property, plant and equipment' or 'capital expenditures' in the context.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}
