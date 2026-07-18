from langgraph.graph import StateGraph, MessagesState, START, END
from pprint import pprint

# def mock_llm(state: MessagesState):
#     return {"messages": [{"role": "ai", "content": "hello world"}]}

# graph = StateGraph(MessagesState)
# graph.add_node(mock_llm)
# graph.add_edge(START, "mock_llm")
# graph.add_edge("mock_llm", END)
# graph = graph.compile()
# result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
# pprint(f"{result}")

from typing import TypedDict
class State(TypedDict):
    number: int
    result: str

def check_even(state: State):
    if state["number"] % 2 == 0:
        return {"result": "Even"}
    return {"result": "Odd"}


def route(state: State):
    if state["number"] < 0:
        return "error"
    return "check"

def error_node(state: State):
    return {"result": "Invalid number"}

from langgraph.graph import StateGraph, START, END
graph = StateGraph(State)

graph.add_node("check", check_even)
graph.add_node("error", error_node)

graph.add_conditional_edges(
    START,
    route,
    {
        "check": "check",
        "error": "error",
    }
)

graph.add_edge("check", END)
graph.add_edge("error", END)

graph = graph.compile()

result = graph.invoke({"number": -10})
print(result)