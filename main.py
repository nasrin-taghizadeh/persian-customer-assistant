import os
import uuid
from data.data_base import load_travel_db
from agent.zeroshot_agent import ZeroShotAgent
from agent.conditional_interrupt import ConditionalInterruptAgent
from agent.add_confirmation import AddConfirmationAgent

local_llm = "llama3:8b"


def set_envs():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Persian Customer Support Bot"
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = "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"
    os.environ['TAVILY_API_KEY'] = "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"


def main():
    data_dir = "data/"
    set_envs()
    travel_db = load_travel_db(data_dir)
    # agent = ZeroShotAgent(local_llm, travel_db, data_dir)
    agent = AddConfirmationAgent(local_llm, travel_db, data_dir)
    agent.define_graph()
    # agent.run_example()

    passenger_id = '3442 587242'
    config = {
        'configurable': {
            'passenger_id': passenger_id,
            'thread_id': str(uuid.uuid4()),
        }
    }
    agent.run("سلام. ساعت پرواز من کی است؟", config)


if __name__ == '__main__':
    main()
