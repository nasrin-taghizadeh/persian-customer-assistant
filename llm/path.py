# Basic paths on my computer, please modify them based on your own!
drive_1 = "/media/nasrin/New Volume/"
drive_2 = "/home/nasrin/"
root = "course/GenAI/huggingface/"
llama_cpp_root = "course/GenAI/llama.cpp"


def get_path_of_nvidia_llama_3_8b():
    return "nvidia/Llama3-ChatQA-1.5-8B", drive_1 + root + "nvidia/Llama3-ChatQA-1.5-8B/"


def get_path_of_meta_llama_3_8b():
    return "meta-llama/Meta-Llama-3-8B", drive_1 + root + "meta-llama/Meta-Llama-3-8B/"


def get_path_of_meta_llama_3_8b_instruct():
    return "meta-llama/Meta-Llama-3-8B-Instruct", drive_1 + root + "meta-llama/Meta-Llama-3-8B-Instruct/"


def get_path_of_meta_llama3_8b_instruct():
    return drive_1 + root + "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"


def get_path_of_llama3_8b_function_call():
    return drive_1 + root + "bartowski/llama-3-8B-function-calling-GGUF/llama-3-8B-function-calling-Q8_0.gguf"


def get_path_of_llama_cpp_server():
    return drive_1 + llama_cpp_root

def get_path_of_meta_llama_31_8b_instruct():
    return "meta-llama/Meta-Llama-3.1-8B-Instruct/", drive_2 + root + "meta-llama/Meta-Llama-3.1-8B-Instruct/"