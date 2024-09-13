from prompt.llama3_quantize_prompt import *
from prompt.nvidia_llama3_prompt import *
from prompt.prompt_utils import *
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from local_llm import *
from tools.utility import *

en_queries = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]
queries = [
    # "سلام. چه طوری؟.",
    "سلام. اطلاعات مربوط به پرواز خودم رو میخواستم دریافت کنم.",
    "قوانین کنسلی بلیط چه طوری هست",
    "میخواهم پروازم را کنسل کنم",
    "برای اینکه پروازم را کنسل کنم چه کار باید بکنم",
    "محل اقامت یا مسافرخونه میخواهم بگیرم",
    "برای تردد داخل شهر مقصدم یک ماشین میخواهم کرایه کنم",
    "سوغاتی چی بخرم",
    "فستیوال موسیقی چی پیشنهاد میدهی",
    "ساعت پرواز منو تغییر بده به فردا ۱۸ بعدازظهر",
    "پرواز منو تغییر بده به شماره ۱۸۱۸"
]


def run_client(client, message):
    if message is None:
        message = "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": message},
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ]
    )

    return completion.choices[0].message


def test():
    template = """Question: {question}

        Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate.from_template(template)
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    question = """
        Question: A rap battle between Stephen Colbert and John Oliver
        """
    llm.invoke(question)

    question = """
        Question: A rap battle between Stephen Colbert and John Oliver
        """
    llm.invoke(question)

    question = """
        Question: A rap battle between Stephen Colbert and John Oliver
        """
    llm.invoke(question)


def test_template():
    # llm, embedding_model = run_llama_3_on_llamacpp_server()
    llm, embedding_model = get_llama_3_8b_llm_instruct_quantize()

    prompt_template = """
        <|im_start|>system {system_prompt}<|im_end|>
        <|im_start|>user {user_prompt}<|im_end|>
        <|im_start|>assistant
        """

    p1 = """
        You are a helpful Persian customer support assistant for Alibaba group. 
        Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
        When searching, be persistent.Expand your query bounds if the first search returns no results.
        If a search comes up empty, expand your search before giving up.
        You are going to have a conversation with two users. The first user is the MAIN USER, ho asks questions and needs to be assisted. The second user is our TOOL MANAGER, which runs the requested tools and delivers the tool results.
        You have access to the following tools to get more information if needed: 
        {tool_descs}
        You must respond ONLY with the JSON schema with the following structure:\n
        {{
            \"THOUGHT\": \"<you should always think about what to do>\",
            \"ACTION\": \"<the action to take, must be one tool_name from above tools>\",
            \"ACTION_PARAMS\": \"<the input parameters to the ACTION, it must be in json format complying with the tool_params>\",
            \"FINAL_ANSWER\": \"<a text containing the final answer to the original input question>\"
        }}\n
        Do NOT add anything before or after the json response.
        Current user: 3442 587242\n
        Current time: 20 July 2024\n
        """

    p2 = "سلام. اطلاعات مربوط به پروازم رو میخواستم دریافت کنم."
    p3 = """
                <function-definitions>
                tool_name -> tavily_search_results_json
                tool_params -> query: string (search query to look up)
                tool_description ->
                A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.

                tool_name -> fetch_user_flight_information
                tool_params -> 
                tool_description ->
                Fetch all tickets for the user along with corresponding flight information and seat assignments. 

                </function-definitions>
        """
    p4 = "<function-definitions>\n" + tool_desc_2 + "\n</function-definitions>"
    prompt = prompt_template.format(system_prompt=p1, user_prompt=p2)
    prompt = prompt.format(tool_descs=p4)

    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)


def test_template_2():
    llm, embedding_model = get_llama_3_8b_llm_instruct_quantize()
    # assistant_prompt = llama3_bartowski_instruct_prompt.partial(
    #     #tool_descs="<function-definitions>\n" + tool_desc + "\n</function-definitions>",
    #     tool_descs=tool_desc_opeanai,
    #     # time=datetime.now()
    # )
    assistant_prompt = tool_selector_prompt.partial(tool_desc=tool_group_desc_2)
    assistant_runnable = assistant_prompt | llm

    queries = [
        "سلام. چه طوری؟.",
        "سلام. اطلاعات مربوط به پرواز خودم رو میخواستم دریافت کنم.",
        "قوانین کنسلی بلیط چه طوری هست",
        "میخواهم پروازم را کنسل کنم",
        "برای اینکه پروازم را کنسل کنم چه کار باید بکنم",
        "محل اقامت یا مسافرخونه میخواهم بگیرم",
        "برای تردد داخل شهر مقصدم یک ماشین میخواهم کرایه کنم",
        "سوغاتی چی بخرم",
        "فستیوال موسیقی چی پیشنهاد میدهی",
    ]
    for query in queries:
        result = assistant_runnable.invoke({"messages": query, "user_info": '3442 587242'})
        print(query, "\n", result)


def test_template_3():
    llm, embedding_model = get_llama_3_8b_llm_instruct_quantize()
    # llm, embedding_model = get_llama_3_on_ollama_server()
    # three calls
    translate_runnable = translate_prompt | llm
    assistant_prompt = tool_selector_prompt.partial(tool_desc=tool_group_desc_2)
    assistant_runnable = assistant_prompt | llm
    for query in queries:
        # first call, translate
        translated_query = translate_runnable.invoke({"messages": query})
        if "Here is the translation:" in translated_query:
            translated_query = translated_query[translated_query.index("Here is the translation:"):].strip()
        print(query, "\n", translated_query)

        # second call, choose tool
        result = assistant_runnable.invoke({"messages": translated_query, "user_info": '3442 587242'})
        print(query, "\n", result)
        tool_name = extract_tool_name(result, "TOOL")
        print("This tool was selected: ", tool_name)

        if tool_name in tool_params.keys():
            # third call, tool param
            tool_prompt = tool_calling_prompt.partial(funct_desc=json.dumps(tool_params[tool_name], indent=2))
            assistant_tool = tool_prompt | llm
            result = assistant_tool.invoke({"messages": query, "user_info": '3442 587242', "time": datetime.now()})
            print(query, "\n", result)
        else:
            print("None or Wrong tool name: ", tool_name)


def test_template_4():
    llm, embedding_model = get_meta_llama_3_8b_on_hf_pipeline("nvidia")

    # three calls
    translate_runnable = nvidia_translate_prompt | llm
    assistant_prompt = nvidia_tool_selector_prompt.partial(tool_desc=tool_group_desc_2)
    assistant_runnable = assistant_prompt | llm
    for query in queries:
        # first call, translate
        translated_query = translate_runnable.invoke({"messages": query})
        translated_query = translated_query[translated_query.index("Assistant:") + len("Assistant:"):].strip()
        print(query, ":", translated_query)
        print("-------------")

        # second call, choose tool
        result = assistant_runnable.invoke({"messages": translated_query, "user_info": '3442 587242'})
        print(result)
        print("++++++++++++++")
        tool_name = extract_tool_name(result, "TOOL")

        if tool_name in tool_params.keys():
            # third call, tool param
            tool_prompt = nvidia_tool_selector_prompt.partial(funct_desc=json.dumps(tool_params[tool_name], indent=2))
            assistant_tool = tool_prompt | llm
            result = assistant_tool.invoke({"messages": query, "user_info": '3442 587242', "time": datetime.now()})
            print(result)
            print("|||||||||||||||||||||||||||||")
        else:
            print("None or Wrong tool name: ", tool_name)


def test_template_5():
    llm, embedding_model = get_meta_llama_3_8b_on_hf_pipeline("nvidia")
    assistant_prompt = nvidia_tool_selector_prompt.partial(tool_desc=tool_desc_python)
    assistant_runnable = assistant_prompt | llm
    messages = ""
    for query in en_queries[:2]:
        print(query)
        messages += f"{query}\n\n"
        result = assistant_runnable.invoke({"messages": messages, "user_info": '3442 587242'})
        assistant_message = result[result.rindex("Assistant:"):]
        messages += assistant_message.strip() + "\n\n"
        print(assistant_message)
        print("++++++++++++++")


if __name__ == "__main__":
    # test_template()
    # test_template_2()
    test_template_3()
    # test_template_4()
    # test_template_5()

    # llm, embed = get_meta_llama_3_8b_on_hf_pipeline("")
    # print(llm.invoke("Hey how are you doing today?"))

    # os.environ["HF_TOKEN"] = "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"
    # llm = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    #     task="text-generation",
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    # )
    # messages = [
    #     SystemMessage(content="You're a helpful assistant"),
    #     HumanMessage(
    #         content="What happens when an unstoppable force meets an immovable object?"
    #     ),
    # ]
    # chat_model = ChatHuggingFace(llm=llm)
    # print(chat_model.model_id)
    # chat_model._to_chat_prompt(messages)
    # res = chat_model.invoke(messages)
    # print(res.content)

    # get_llama_3_8b_on_hf_endpoint()
