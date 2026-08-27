"""

LangChain agent.


"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent.classifier import classify as _classify

load_dotenv()

MODEL = os.environ.get("WAFER_AGENT_MODEL", "anthropic:claude-sonnet-5")


@tool(description=("Classify  and analyze a wafer map with the trained CNN. "))
def classify_wafer_map(wafer_map: str) -> dict:
    return _classify(wafer_map)


SYSTEM_PROMPT = """You are a semiconductor process engineer working with WM-811K
wafer maps.

When a wafer map is given, call classify_wafer_map on it, then reply
with:

1. The predicted defect type and its confidence.
2. What the confidence mean here, and what the consequences are for the process. If the prediction is uncertain, mention the runner-up class.

Points 2  come from your own knowledge of semiconductor manufacturing. 

State the confidence plainly. If it is below about 0.6, say the prediction is
uncertain and mention the runner-up class.

For follow-up questions, reuse the classification already in the conversation
instead of re-running the classifier. Write in plain language for an user
who may not know the model internals."""


_agent = None
_checkpointer = None


def get_agent():
    global _agent, _checkpointer

    if _agent is None:
        _checkpointer = InMemorySaver()
        _agent = create_agent(
            model=MODEL,
            tools=[classify_wafer_map],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_checkpointer,
        )

    return _agent


def ask(message: str, session_id: str) -> str:
    agent = get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": session_id}},
    )
    return result["messages"][-1].text


if __name__ == "__main__":
    pass
