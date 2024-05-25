import shutil
import uuid
from typing import Annotated
from tools.flight import *
from tools.hotel import *
from tools.excursion import *
from tools.car_rental import *
from tools.lookup_policy import *
from tools.utility import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import AnyMessage, add_messages
from data.data_base import backup_file
from tools.utility import _print_event


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class ConditionalInterruptAgent():
    def __init__(self, local_llm, db, data_dir):
        self.db = db
        # Haiku is faster and cheaper, but less accurate
        # llm = ChatAnthropic(model="claude-3-haiku-20240307")
        # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
        # You could also use OpenAI or another model, though you will likely have
        # to adapt the prompts
        # from langchain_openai import ChatOpenAI

        # llm = ChatOpenAI(model="gpt-4-turbo-preview")
        #######################
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key="ollama",
            model=local_llm,
            base_url="http://localhost:11434/v1")
        #####################
        self.llm = llm
        assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for AliBaba group. "
                    " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    " Answer in Persian not English. "
                    "\n\nCurrent user:\n\n{user_info}\n"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

        self.part_2_tools = [
            TavilySearchResults(max_results=1),
            FetchUserFlightInfo(db),
            SearchFlight(db),
            LookupPolicy(local_llm, data_dir),
            UpdateTicket(db),
            CancelTicket(db),
            SearchCarRentals(db),
            BookCarRental(db),
            UpdateCarRental(db),
            CancelCarRental(db),
            SearchHotel(db),
            BookHotel(db),
            UpdateHotel(db),
            CancelHotel(db),
            SearchTrip(db),
            BookExcursion(db),
            UpdateExcursion(db),
            CancelExcursion(db),
        ]
        self.part_2_assistant_runnable = assistant_prompt | llm.bind_tools(self.part_2_tools)

    def define_graph(self):
        builder = StateGraph(State)
        # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
        # having to take an action
        builder.add_node("fetch_user_info", self.user_info)
        builder.set_entry_point("fetch_user_info")
        builder.add_node("assistant", Assistant(self.part_2_assistant_runnable))
        builder.add_node("action", create_tool_node_with_fallback(self.part_2_tools))
        builder.add_edge("fetch_user_info", "assistant")
        builder.add_conditional_edges(
            "assistant", tools_condition, {"action": "action", END: END}
        )
        builder.add_edge("action", "assistant")

        memory = SqliteSaver.from_conn_string(":memory:")
        self.part_2_graph = builder.compile(
            checkpointer=memory,
            # NEW: The graph will always halt before executing the "action" node.
            # The user can approve or reject (or even alter the request) before
            # the assistant continues
            interrupt_before=["action"],
        )

    def user_info(self, state: State):
        return {"user_info": FetchUserFlightInfo(self.db).invoke({})}

    def run_example(self):
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
        tutorial_questions = [
            "سلام. ساعت پرواز من کی است؟",
            "میتوانم پروازم را عوض کنم؟ زودتر بگیرم؟",
            "پرواز من را عوض کن. بگذار هفته بعد.",
            "آیا هتل خاصی را پیشنهاد میکنی؟",
            "جایی را برای گشت و گذار پیشنهاد میکنی",
            "ممنون"
        ]
        _printed = set()
        # We can reuse the tutorial questions from part 1 to see how it does.
        for question in tutorial_questions:
            events = self.part_2_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)
            snapshot = self.part_2_graph.get_state(config)
            while snapshot.next:
                # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
                # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
                # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
                if user_input.strip() == "y":
                    # Just continue
                    result = self.part_2_graph.invoke(
                        None,
                        config,
                    )
                else:
                    # Satisfy the tool invocation by
                    # providing instructions on the requested changes / change of mind
                    result = self.part_2_graph.invoke(
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
                snapshot = self.part_2_graph.get_state(config)
