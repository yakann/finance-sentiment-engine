from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    messages: list[BaseMessage]


def greet(state: State) -> State:
    last = state["messages"][-1]
    reply = AIMessage(content=f"Hello! I received: '{last.content}'")
    return {"messages": state["messages"] + [reply]}


def reply_node(state: State) -> State:
    response = AIMessage(content="Graph traversal complete. Have a great day!")
    return {"messages": state["messages"] + [response]}


def build_graph() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("greet", greet)
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", "reply")
    builder.add_edge("reply", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke({"messages": [HumanMessage(content="LangGraph day 22!")]})

    print("=== Graph Output ===")
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"[{role}] {msg.content}")

    # PNG export
    png_bytes = graph.get_graph().draw_mermaid_png()
    output_path = "graph/hello_graph.png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"\nGraph PNG saved → {output_path}")
