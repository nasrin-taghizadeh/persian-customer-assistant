from langchain.chains import LLMChain
# from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5Tokenizer, T5ForConditionalGeneration, GPT2TokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

model_path = "/media/nasrin/New Volume/course/GenAI/huggingface/meta-llama__Meta-Llama-3-8B/"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=False) # set load_in_8bit=True to use GPU
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('making a pipeline...')
#max_length has typically been deprecated for max_new_tokens
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, model_kwargs={"temperature":0}
)
hf = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, llm=hf)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))
