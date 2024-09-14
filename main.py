import os
import uuid
from data.data_base import load_travel_db
from assistant.assistant_with_three_agents import AliBabaAssistant
from llm.local_llm import *

local_llm = "llama3:8b"


def set_envs():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Persian Customer Support Bot"
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = ""
    os.environ['TAVILY_API_KEY'] = ""
    os.environ['GROQ_API_KEY'] = ""
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    os.environ["HF_TOKEN"] = ""


def main():
    data_dir = "data/"
    set_envs()
    travel_db = load_travel_db(data_dir)

    # llm, embedding_model = get_llama_3_8b_on_hf_hub("meta")
    # llm, embedding_model = get_meta_llama_3_8b_on_hf_pipeline()
    # llm, embedding_model = run_llama_3_on_llamacpp_server()
    # llm, embedding_model = get_llama_3_8b_llm_instruct_quantize_on_chatollama()

    assistant = AliBabaAssistant(travel_db, data_dir)
    assistant.define_graph()
    # assistant.run_example()

    passenger_id = '3442 587242'
    config = {
        'configurable': {
            'passenger_id': passenger_id,
            'thread_id': str(uuid.uuid4()),
        }
    }
    assistant.run("Hi there, what time is my flight?", config)
    assistant.run("Am i allowed to update my flight to something sooner? I want to leave later today.", config)
    assistant.run("Update my flight to sometime next week then", config)
    assistant.run("The next available option is great", config)

    # assistant.run("سلام. ساعت پرواز من کی است؟", config)
    # assistant.run("سلام. اطلاعات مربوط به پروازم رو میخواستم دریافت کنم.", config)
    # assistant.run("قوانین کنسلی بلیط چه طوری هست", config)


if __name__ == '__main__':
    main()
