import shutil
import uuid
import json
import warnings
from typing import Annotated, Literal, List
from tools.utility import *

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolCall, AIMessage
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import AnyMessage, add_messages
from data.data_base import backup_file
from tools.utility import _print_event


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class TranslateAgent:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig, ):
        # getting output of LLM
        result = self.runnable.invoke(state, config)
        messages = state["messages"] + ["User:\n" + result]
        state = {**state, "messages": messages}
        return state


class ToolSelectorAgent:
    def __init__(self, runnable: Runnable, tools: List[BaseTool]):
        self.runnable = runnable
        self.tools = tools

    def __call__(self, state: State, config: RunnableConfig, ):
        action = None
        while True:
            # getting output of LLM
            passenger_id = config['configurable'].get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state, config)
            try:
                # check whether output is a valid json file, i.e. parsable to json
                if '{' in result and '}' in result:
                    start_index = result.index('{')
                    end_index = result.rindex('}')
                    result = result[start_index:end_index + 1]
                content_json = json.loads(result)
            except ValueError as e:
                # in case of invalid json file, ask llm to respond with a valid json.
                warnings.warn('Invalid json Format: ' + result)
                # state['messages'] += [result, HumanMessage("Respond with a valid json output!")]
                state['messages'] += [result, "User: Respond with a valid json output!"]
                continue

            selected_tool = content_json.get('TOOL', None)
            if selected_tool is None:
                answer = content_json.get("ANSWER", "")
                state['messages'] += [result, f"Assistant:\n`{answer}`"]
                return state

            if selected_tool not in [tool.name for tool in self.tools]:
                warnings.warn('BAD TOOL NAME: ' + selected_tool)
                # state['messages'] += [result, HumanMessage(f"The Tool `{tool}` does not exist!")]
                state['messages'] += [result, f"The Tool `{selected_tool}` does not exist!"]
                continue

            break
        return {'messages': [result, selected_tool]}


class ToolCallingAgent:
    def __init__(self, runnable: Runnable, tools: List[BaseTool]):
        self.runnable = runnable
        self.tools = tools

    def __call__(self, state: State, config: RunnableConfig, ):
        action = None
        while True:
            # getting output of LLM
            selected_tool = state["messages"][-1]
            tool_desc = get_tool_openai_format(selected_tool, self.tools)
            passenger_id = config['configurable'].get("passenger_id", None)
            state = {**state, "user_info": passenger_id, "funct_desc": tool_desc}
            result = self.runnable.invoke(state, config)
            try:
                # check whether output is a valid json file, i.e. parsable to json
                if '{' in result and '}' in result:
                    start_index = result.index('{')
                    end_index = result.rindex('}')
                    result = result[start_index:end_index + 1]
                content_json = json.loads(result)
            except ValueError as e:
                # in case of invalid json file, ask llm to respond with a valid json.
                warnings.warn('Invalid json Format: ' + result)
                # state['messages'] += [result, HumanMessage("Respond with a valid json output!")]
                state['messages'] += [result, "Respond with a valid json output!"]
                continue

            tool_name = content_json.get('name', '').replace(' ', '')
            tool_params = content_json.get('arguments') or {}
            if type(tool_params) is str:
                tool_params = json.loads(tool_params)
            answer = content_json.get('ANSWER')

            if action and action not in [tool.name for tool in self.tools]:
                warnings.warn('BAD TOOL NAME: ' + result)
                # state['messages'] += [result, HumanMessage(f"The ACTION `{action}` does not exist!")]
                state['messages'] += [result, f"The ACTION `{action}` does not exist!"]
                continue

            break

        if action:
            tool_call = ToolCall(name=tool_name, args=tool_params, id=str(uuid.uuid4()))
            result.tool_calls.append(tool_call)
            return {'messages': result}

        # final_result = AIMessage(final_answer)
        # return {'messages': [result, final_result]}
        return {'messages': [result, answer]}


class AliBabaAssistant():
    def __init__(self, db, data_dir, translate_prompt, tool_selector_prompt, tool_calling_prompt, llm, embedding_model):
        self.db = db
        self.llm = llm

        # "Read"-only tools (such as retrievers) don't need a user confirmation to use
        self.safe_tools = get_safe_tools(data_dir, db, embedding_model)

        # These tools all change the user's reservations.
        # The user has the right to control what decisions are made
        self.sensitive_tools = get_sensitive_tools(db)
        self.sensitive_tool_names = {t.name for t in self.sensitive_tools}
        tool_selector_prompt = tool_selector_prompt.partial(
            #tool_descs=get_tool_group_description(self.safe_tools + self.sensitive_tools)
            tool_descs =build_funccall_prompt(self.safe_tools + self.sensitive_tools)
        )
        self.tool_selector_assistant_runnable = tool_selector_prompt | llm
        tool_calling_prompt = tool_calling_prompt.partial(time=datetime.now())
        self.tool_calling_assistant_runnable = tool_calling_prompt | llm
        self.translate_assistant_runnable = translate_prompt | llm

    def route_tools(self, state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call = ai_message.tool_calls[0]
        if first_tool_call["name"] in self.sensitive_tool_names:
            return "sensitive_tools"
        return "safe_tools"

    def define_graph(self):
        builder = StateGraph(State)
        # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
        # having to take an action
        builder.add_node("fetch_user_info", self.user_info)
        builder.set_entry_point("fetch_user_info")
        builder.add_node("translate_assistant", TranslateAgent(self.translate_assistant_runnable))
        builder.add_node("tool_selection_assistant", ToolSelectorAgent(self.tool_selector_assistant_runnable,
                                                                       self.safe_tools + self.sensitive_tools))
        builder.add_node("tool_calling_assistant", ToolCallingAgent(self.tool_calling_assistant_runnable,
                                                                    self.safe_tools + self.sensitive_tools))
        builder.add_node("safe_tools", create_tool_node_with_fallback(self.safe_tools))
        builder.add_node("sensitive_tools", create_tool_node_with_fallback(self.sensitive_tools))

        # Define logic
        builder.add_edge("fetch_user_info", "translate_assistant")
        builder.add_edge("fetch_user_info", "tool_selection_assistant")
        builder.add_edge("fetch_user_info", "tool_calling_assistant")
        builder.add_edge("translate_assistant", "tool_selection_assistant")
        builder.add_edge("tool_selection_assistant", "tool_calling_assistant")
        builder.add_conditional_edges("tool_calling_assistant", self.route_tools)
        builder.add_edge("safe_tools", "tool_calling_assistant")
        builder.add_edge("sensitive_tools", "tool_calling_assistant")

        memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = builder.compile(
            checkpointer=memory,
            # NEW: The graph will always halt before executing the "action" node.
            # The user can approve or reject (or even alter the request) before
            # the assistant continues
            interrupt_before=["sensitive_tools"],
        )

    def user_info(self, state: State):
        return {"user_info": FetchUserFlightInfo(self.db).invoke({})}

    def run_example(self, tutorial_questions):
        # Update with the backup file so we can restart from the original place in each section
        shutil.copy(backup_file, self.db)
        thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                # The passenger_id is used in our flight tools to
                # fetch the user's flight information
                "passenger_id": "3442 587242",
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            }
        }
        _printed = set()
        # We can reuse the tutorial questions from part 1 to see how it does.
        for question in tutorial_questions:
            events = self.graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)
            snapshot = self.graph.get_state(config)
            while snapshot.next:
                # We have an interrupt! The assistant is trying to use a tool, and the user can approve or deny it
                # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
                # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
                if user_input.strip() == "y":
                    # Just continue
                    result = self.graph.invoke(
                        None,
                        config,
                    )
                else:
                    # Satisfy the tool invocation by
                    # providing instructions on the requested changes / change of mind
                    result = self.graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                    content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )
                snapshot = self.graph.get_state(config)

    def run(self, question, config):
        _printed = set()
        events = self.graph.stream(
            #  {'messages': [HumanMessage(question)]}, config, stream_mode='values'
            {'messages': [question]}, config, stream_mode='values'
        )
        for event in events:
            _print_event(event, _printed)
        snapshot = self.graph.get_state(config)
        while snapshot.next:
            # We have an interrupt! The assistant is trying to use a tool, and the user can approve or deny it
            # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
            # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
            if user_input.strip() == "y":
                # Just continue
                result = self.graph.invoke(
                    None,
                    config,
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = self.graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = self.graph.get_state(config)
