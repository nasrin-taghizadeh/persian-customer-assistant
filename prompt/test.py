from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder, PromptTemplate,
)

# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful AI bot. Your name is {name}."),
#         ("human", "Hello, how are you doing?"),
#         ("ai", "I'm doing well, thanks!"),
#         ("human", "{user_input}"),
#     ]
# )
#
# messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
# print(messages)


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(messages)
print("----------")

from langchain_core.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
chat_message_prompt = chat_message_prompt.format(subject="force")
print(chat_message_prompt)
print("----------")

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

chat_prompt = chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
)
print(chat_prompt.to_messages())
print("---")
print(chat_prompt.to_string())
print("----------")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)