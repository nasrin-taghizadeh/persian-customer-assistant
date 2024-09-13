# from IPython.display import Image, display
import os
import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
from langchain_community.llms.llamacpp import LlamaCpp
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ['TAVILY_API_KEY'] = "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"
tool = TavilySearchResults(max_results=2)
tools = [tool]


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


llama_model_path = "/media/nasrin/New Volume/course/GenAI/huggingface/bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
llm = LlamaCpp(
    model_path=llama_model_path,
    temperature=0.0,
    max_tokens=1000,
    top_p=1,
    top_k=0.1,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=8000,
    n_gpu_layers=32,
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    # print("=== user input ===")
    # print(user_input)
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1])
