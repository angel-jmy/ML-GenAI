# .env:OPENAI_API_KEY = "..."
from dotenv import load_dotenv
load_dotenv()

# from openai import OpenAI
# client = OpenAI()

# response = client.completions.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt="You are a helpful assistant,Write a tagline for an ice cream shop."
# )

# print(response.choices[0].text.strip())


# Chat Model

# from openai import OpenAI
# client = OpenAI()

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ]
# )

# print(response.choices[0].message.content)


"""
pip install langchain
"""

# LangChain text model

# from langchain_openai import OpenAI, ChatOpenAI

# llm = OpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.8,
#     max_tokens=60,
# )


# response = llm.predict("Write a tagline for an ice cream shop.")
# print(response)

# from langchain_openai import OpenAI

# Text Model
# llm = OpenAI(
#     model="gpt-3.5-turbo-instruct",
#     temperature=0.5,
#     max_tokens=50
# )

# response = llm.predict("Write a tagline for an ice cream shop.")
# print(response)


# LangChain chat model

# from langchain_community.chat_models import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4", temperature=0)
# response = llm.predict("Write a tagline for an ice cream shop.")
# print(response)


from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-4", temperature=0)

message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Write a tagline for an ice cream shop.")
]

response = chat(message)
print(response.content)
