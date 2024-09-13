import os
import subprocess
import threading
import openai
import transformers
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import OllamaEmbeddings, LlamaCppEmbeddings
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, LlamaTokenizer
from transformers import LlamaForCausalLM
from llm.path import *

import torch

print(torch.cuda.is_available())

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def get_meta_llama_3_8b_on_hf_pipeline(model_name=""):
    """
    When LLM is in safe-tensor format on local machine
    :param model_name:
    :return: llm
    """
    if model_name == "":
        repo_name, model_path = get_path_of_meta_llama_3_8b()
        print("LLama model is loading...")
    elif model_name == "instruct":
        # repo_name, model_path = get_path_of_meta_llama_3_8b_instruct()
        repo_name, model_path = get_path_of_meta_llama_31_8b_instruct()
        print("LLama 3.1 instruct model is loading...")
    elif model_name == "nvidia":
        repo_name, model_path = get_path_of_nvidia_llama_3_8b()
        print("Nvidia QA model is loading...")
    else:
        return None, None

    config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", # quantization_config=config,
                                             use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print('making a pipeline...')
    # max_length has typically been deprecated for max_new_tokens
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, model_kwargs={"temperature": 0}
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    embedding_model = None
    # embedding_model = HuggingFaceEmbeddings(model_name=model_path)
    return llm, embedding_model


def get_llama_3_8b_on_hf_endpoint():
    """
    when LLM is fully run on the HF, not on the local pc
    :return: llm available on HF
    """
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content="What happens when an unstoppable force meets an immovable object?"
        ),
    ]
    chat_model = ChatHuggingFace(llm=llm)
    print(chat_model.model_id)
    chat_model._to_chat_prompt(messages)
    res = chat_model.invoke(messages)
    print(res.content)


def get_llama_3_8b_llm_instruct_quantize(temp=0.0, max_tok=2000, top_p=1, top_k=0.1):
    llama_model_path = get_path_of_meta_llama3_8b_instruct()
    # llama_model_path = get_path_of_llama3_8b_function_call()
    llm = LlamaCpp(
        model_path=llama_model_path,
        temperature=temp,
        max_tokens=max_tok,
        top_p=top_p,
        top_k=top_k,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=8000,
        n_gpu_layers=32,

    )
    embedding_model = None
    # embedding_model = LlamaCppEmbeddings(model_path=llama_model_path)
    # llm.create_embedding()
    return llm, embedding_model


def get_llama_3_8b_llm_instruct_quantize_on_chatollama():
    # llama_model_path = get_path_of_meta_llama3_8b_instruct()
    # llama_model_path = get_path_of_llama3_8b_function_call()
    llama_model_path = get_path_of_meta_llama_3_8b()

    bashCommand = "ollama pull {}".format(llama_model_path)

    def basher():
        subprocess.call(bashCommand, shell=True)

    t = threading.Thread(target=basher)
    t.start()

    llm = ChatOllama(url="", format="json", temperature=0)
    # embedding_model = LlamaCppEmbeddings(model_path=llama_model_path)
    # llm.create_embedding()
    embedding_model = None
    return llm, embedding_model


def get_llama_3_on_ollama_server():
    local_llm = "llama3:8b"
    llama_model_path = get_path_of_meta_llama_3_8b()
    bashCommand = "ollama pull {}".format(llama_model_path)

    def basher():
        subprocess.call(bashCommand, shell=True)

    t = threading.Thread(target=basher)
    t.start()
    llm = ChatOpenAI(
        api_key="ollama",
        model=local_llm,
        base_url="http://localhost:11434/v1")
    embedding_model = OllamaEmbeddings(model=local_llm, num_thread=8, temperature=0.0)
    return llm, embedding_model


def run_llama_3_on_llamacpp_server(port=8080):
    root = get_path_of_llama_cpp_server()
    root = root.replace(" ", "\ ")
    model_path = get_path_of_meta_llama3_8b_instruct()
    gpu_layers = 10
    bashCommand = "{}/llama-cli -m {} -c 2048 --port {} -ngl {}".format(root, model_path, port, gpu_layers)

    def basher():
        subprocess.call(bashCommand, shell=True)

    t = threading.Thread(target=basher)
    t.start()
    # os.system(bashCommand)
    url = "http://localhost:{}/completion".format(port)

    llm = ChatOpenAI(
        api_key="sk-no-key-required",
        model="llama3",
        base_url=url
    )
    # embedding_model = OllamaEmbeddings(model="llama3", num_thread=8, temperature=0.0, base_url=url)
    embedding_model = None
    return llm, embedding_model


def get_llama3_as_openai_client():
    client = openai.OpenAI(
        base_url="http://localhost:8080/completion",  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required"
    )
    return client


def get_llama_3_with_transformers():
    # model_id = "meta-llama/Meta-Llama-3-8B"
    model_id = get_path_of_meta_llama_3_8b()
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    pipeline("Hey how are you doing today?")
    return pipeline
