import json
import inspect
from typing import List
from docstring_extractor import get_docstrings
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

from tools.flight import *
from tools.hotel import *
from tools.excursion import *
from tools.car_rental import *
from tools.lookup_policy import *


def get_safe_tools(data_dir, db, embedding_model):
    return [
        TavilySearchResults(max_results=1),
        FetchUserFlightInfo(db),
        SearchFlight(db),
        SearchCarRentals(db),
        LookupPolicy(data_dir, embedding_model),
        SearchHotel(db),
        SearchTrip(db),
    ]


def get_sensitive_tools(db):
    return [
        UpdateTicket(db),
        CancelTicket(db),
        BookCarRental(db),
        UpdateCarRental(db),
        CancelCarRental(db),
        BookHotel(db),
        UpdateHotel(db),
        CancelHotel(db),
        BookExcursion(db),
        UpdateExcursion(db),
        CancelExcursion(db),
    ]


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    """
    creates a node for calling tools
    :param tools: list of available tools to the assistant
    :return: a node in the graph responsible for calling tools
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def format_tool_to_3lines(tool: BaseTool) -> str:
    tool_params = []
    for name, info in tool.args.items():
        if 'type' in info:
            tool_params.append(f"{name}: {info['type']} ({info['description']})")
        elif 'anyOf' in info:
            anyoftype = ", ".join([anytype['type'] for anytype in info['anyOf']])
            tool_params.append(f"{name}: any of {anyoftype} ({info['description']})")

    tool_params_string = ', '.join(tool_params)
    return (
        f"tool_name -> {tool.name}\n"
        f"tool_params -> {tool_params_string}\n"
        f"tool_description -> {tool.description}"
    )


# def format_tool_to_bullet(tool: BaseTool) -> str:
#     tool_params = []
#     '- list_tvshows: List of TV Shows in the database. Example: {"function":"list_tvshows"}</s> returns [{"id":3948, "name":"Doctor who"},{"id":12, "name":"Brooklynn Nine-Nine"}]'
#     for name, info in tool.args.items():
#         if 'type' in info:
#             tool_params.append(f"{name}: {info['type']} ({info['description']})")
#         elif 'anyOf' in info:
#             anyoftype = ", ".join([anytype['type'] for anytype in info['anyOf']])
#             tool_params.append(f"{name}: any of {anyoftype} ({info['description']})")
#
#     tool_params_string = ', '.join(tool_params)
#     return (
#         f"- {tool.name}: {tool.description}. Example: {} -> {tool_params_string}\n"
#     )
#
# def format_tool_to_html_tag(tool: BaseTool) -> str:
#     return ""
#
#
# def get_tools_description_tag_format(tools: List[BaseTool]) -> str:
#     tool_desc = [format_tool_to_3lines(tool) for tool in tools]
#     return "<function-definitions>\n" + tool_desc + "\n</function-definitions>"

def get_all_tools_openai_format(tools: List[BaseTool]) -> str:
    tool_desc = [format_tool_to_openai_function(tool) for tool in tools]
    # tool_desc_dict = {tool.name: format_tool_to_openai_function(tool) for tool in tools}
    # print(json.dumps(tool_desc_dict, indent=2))
    return json.dumps(tool_desc, indent=2)


def get_tool_openai_format(tool_name: str, all_tools: List[BaseTool]) -> str:
    for tool in all_tools:
        if tool.name == tool_name:
            tool_desc = format_tool_to_openai_function(tool)
            return json.dumps(tool_desc, indent=2)
    return ""


def get_tool_group_description(tools: dict) -> str:
    tool_desc = []
    for i, tool in enumerate(tools):
        try:
            tool_desc.append(
                f"{i + 1}: Tool name: {tool.name}, Tool description: {tool.get_file_docstring()}"
            )
        except:
            if tool.__doc__ is not None:
                tool_desc.append(tool.__doc__)

    return "\n".join(tool_desc)


def extract_tool_name(text: str, key):
    try:
        return json.loads(text)[key]
    except:
        if "\"" + key + "\"" in text:
            try:
                index_1 = text.index("\"" + key + "\"") + len(key) + 3
                index_2 = text[index_1:].index(",")
                return text[index_1: index_1 + index_2].strip()[1:-1]
            except:
                return None
        else:
            return None


# def translate_text(target: str, text: str) -> dict:
#     """Translates text into the target language.
#
#     Target must be an ISO 639-1 language code.
#     See https://g.co/cloud/translate/v2/translate-reference#supported_languages
#     """
#     from google.cloud import translate_v2 as translate
#
#     translate_client = translate.Client()
#
#     if isinstance(text, bytes):
#         text = text.decode("utf-8")
#
#     # Text can also be a sequence of strings, in which case this method
#     # will return a sequence of results for each text.
#     result = translate_client.translate(text, target_language=target)
#
#     print("Text: {}".format(result["input"]))
#     print("Translation: {}".format(result["translatedText"]))
#     print("Detected source language: {}".format(result["detectedSourceLanguage"]))
#
#     return result


def build_funccall_prompt(tools: dict):
    nvidia_prompt = ""
    for tool in tools:
        tool_params = []
        detailed_params = ""
        for name, info in tool.args.items():
            if 'type' in info:
                tool_params.append(f"{name}: {info['type']}")
                detailed_params += f"\t- {name} ({info['type']}): {info['description']}\n"
            elif 'anyOf' in info:
                anyoftype = "|".join([anytype['type'] for anytype in info['anyOf']])
                tool_params.append(f"{name}: {anyoftype}")
                detailed_params += f"\t- {name} ({anyoftype}): {info['description']}\n"
        signature = ", ".join(tool_params)
        prompt = \
            f'''
def {tool.name}({signature}):
    """
    {tool.description}
    """
    Parameters:
{detailed_params}
'''
        nvidia_prompt += prompt
    print(nvidia_prompt)
    return nvidia_prompt


if __name__ == "__main__":
    # print(translate_text("en", "سلام چه طوری"))
    print(build_funccall_prompt([]))
