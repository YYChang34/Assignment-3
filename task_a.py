from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate

from config import get_llm


def run_legacy_agent(question: str, retrievers: dict) -> str:
    tools = [
        create_retriever_tool(
            retriever,
            f"search_{key}_financials",
            f"Searches and retrieves financial data from {key.capitalize()}'s 10-K annual report (fiscal year 2024).",
        )
        for key, retriever in retrievers.items()
    ]

    if not tools:
        return "System Error: No tools available. Please run 'python build_rag.py' first."

    llm = get_llm()
    template = """You are a precise financial analyst assistant. You have access to the following tools:

{tools}

CRITICAL BEHAVIORAL RULES:
1. [English Only] Your Final Answer MUST always be written in English, even if the user asks in Chinese or another language.
2. [Year Precision] Financial reports contain data for multiple years (2024, 2023, 2022). You MUST report the EXACT fiscal year 2024 figure. Never substitute 2023 or 2022 data unless the question explicitly asks for it.
3. [Honesty] If the exact 2024 figure cannot be found in the retrieved documents, state "I don't know" rather than guessing, estimating, or extrapolating from other years.
4. [Source Citation] Always mention which company's data you are using.

Use the following format EXACTLY:

Question: the input question you must answer
Thought: reason about what information you need and which tool to use
Action: the action to take, must be one of [{tool_names}]
Action Input: the specific financial query to send to the tool
Observation: the result returned by the tool
... (repeat Thought/Action/Action Input/Observation as needed until you find the answer)
Thought: I now know the final answer
Final Answer: the final answer in English with the exact figure and its source

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"
