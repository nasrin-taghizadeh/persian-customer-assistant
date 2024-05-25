import shutil
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from tools.flight import *
from tools.hotel import *
from tools.excursion import *
from tools.car_rental import *
from tools.lookup_policy import *
from tools.utility import *
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from data.data_base import backup_file
from tools.utility import _print_event


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Agent
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            passenger_id = config.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
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
            else:
                break
        return {"messages": result}


class ZeroShotAgent():
    def __init__(self, local_llm, db, data_dir):
        # llm = ChatOllama(model=local_llm, format="json", temperature=1)
        llm = ChatOpenAI(
            api_key="ollama",
            model=local_llm,
            base_url="http://localhost:11434/v1")
        self.llm = llm
        self.db = db
        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for AliBaba group. "
                    " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up. "
                    " Answer in Persian not English. "
                    "\n\nCurrent user:\n\n{user_info}\n"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

        self.part_1_tools = [
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
        self.part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(self.part_1_tools)

    def define_graph(self):

        builder = StateGraph(State)
        # Define nodes: these do the work
        builder.add_node("assistant", Assistant(self.part_1_assistant_runnable))
        builder.add_node("action", create_tool_node_with_fallback(self.part_1_tools))
        # Define edges: these determine how the control flow moves
        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            # "action" calls one of our tools. END causes the graph to terminate (and respond to the user)
            {"action": "action", END: END},
        )
        builder.add_edge("action", "assistant")

        # The checkpointer lets the graph persist its state
        # this is a complete memory for the entire graph.
        memory = SqliteSaver.from_conn_string(":memory:")
        self.part_1_graph = builder.compile(checkpointer=memory)

    def run_example(self):
        # Let's create an example conversation a user might have with the assistant
        tutorial_questions = [
            # "Hi there, what time is my flight?",
            # "Am i allowed to update my flight to something sooner? I want to leave later today.",
            # "Update my flight to sometime next week then",
            # "The next available option is great",
            # "what about lodging and transportation?",
            # "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
            # "OK could you place a reservation for your recommended hotel? It sounds nice.",
            # "yes go ahead and book anything that's moderate expense and has availability.",
            # "Now for a car, what are my options?",
            # "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
            # "Cool so now what recommendations do you have on excursions?",
            # "Are they available while I'm there?",
            # "interesting - i like the museums, what options are there? ",
            # "OK great pick one and book it for my second day there.",
            "سلام. ساعت پرواز من کی است؟",
            "میتوانم پروازم را عوض کنم؟ زودتر بگیرم؟",
            "پرواز من را عوض کن. بگذار هفته بعد.",
            "آیا هتل خاصی را پیشنهاد میکنی؟",
            "جایی را برای گشت و گذار پیشنهاد میکنی",
            "ممنون"
        ]

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
        for question in tutorial_questions:
            events = self.part_1_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)

    def run(self, question, config):
        _printed = set()
        events = self.part_1_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
