import json

from config import get_llm, FILES, retry_logic
from state import AgentState


def build_retrieve_node(retrievers: dict):

    @retry_logic
    def retrieve_node(state: AgentState) -> dict:
        question = state["question"]
        llm = get_llm()

        options = list(FILES.keys()) + ["both", "none"]

        router_prompt = f"""You are a financial document router. Your job is to classify the user's question
and route it to the correct data source.

Available options: {options}

Classification rules:
- "apple": if the question is about Apple, iPhone, Mac, iPad, AAPL, Tim Cook, Apple services, or Apple financials
- "tesla": if the question is about Tesla, EV, TSLA, Elon Musk, Cybertruck, Autopilot, or Tesla financials
- "both": if the question explicitly compares Apple and Tesla, or asks about both companies together
- "none": if the question is about neither Apple nor Tesla

Output ONLY valid JSON with no explanation: {{"datasource": "<one of: {', '.join(options)}>"}}

User Question: {question}"""

        try:
            response = llm.invoke(router_prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            res_json = json.loads(content)
            target = res_json.get("datasource", "both")
            if target not in options:
                target = "both"
        except Exception as e:
            print(f"Error parsing router output: {e}.")
            target = "both"

        docs_content = ""
        targets_to_search = list(FILES.keys()) if target == "both" else ([target] if target in FILES else [])

        for t in targets_to_search:
            if t in retrievers:
                docs = retrievers[t].invoke(question)
                source_name = t.capitalize()
                docs_content += f"\n\n[Source: {source_name} 10-K]\n" + "\n".join([d.page_content for d in docs])

        return {"documents": docs_content, "search_count": state["search_count"] + 1}

    return retrieve_node
