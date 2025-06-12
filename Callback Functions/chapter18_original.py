import asyncio

async def compute(x,y,callback):
    print("starting computer......")
    await asyncio.sleep(0.5)
    result = x + y
    callback(result)
    print("finished computer......")

def print_result(value):
    print(f"the result is :{value}")

async def squared_result():
    print("starting another task ......")
    # print(f"The squared result is: {value**2}")
    await asyncio.sleep(1)
    print("finished another task......")

async def main():
    print("starting ......")
    task1 = asyncio.create_task(compute(3,4,print_result))
    task2 = asyncio.create_task(squared_result())

    await task1
    await task2
    print("ending ....")

asyncio.run(main())


#callback functions
def compute(x,y,callback):
    result = x + y
    callback(result)

def print_result(value):
    print(f"The result is: {value}")

def squared_result(value):
    print(f"The squared result is: {value**2}")
    
compute(3,4,print_result)
compute(3,4,squared_result)
#The squared result is: 49
#The result is: 7

"""
Callback Handler in LangChain
CallbackManager->BaseCallbackHandler

Using Callback Handlers in Components:Chain,Model,Agents,Tools....
    Constructor callbacks: 
    request callbacks: 
"""

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


logfile = "/chapter18/output.log"

logger.add(logfile,colorize=True,enqueue=True)

handler = FileCallbackHandler(logfile)

llm = OpenAI()

prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm,prompt=prompt,callbacks=[handler],verbose=True)

result = chain.run(number=2)

logger.info(result)




"""
Custom callback function
BaseCallbackHandle 
AsyncCallbakHandler
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio#event loop
from typing import Dict, List,Any

from langchain_openai import OpenAI,ChatOpenAI
from langchain.schema import LLMResult,HumanMessage
from langchain.callbacks.base import BaseCallbackHandler,AsyncCallbackHandler

#Create a synchronous callback handler
class FlowerSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"flower data:token:{token}")
#Create a AsyncHandler
class FlowerAsyncHandler(AsyncCallbackHandler):
    async def  on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("flower data!!!!!!!!!!")
        await asyncio.sleep(0.5)
        print("flower data fetched .Providing suggestings!!!!!!!!!!!")
    
    async def on_llm_end(self, response: LLMResult,**kwargs:Any) -> None:
        print("Organizing flower suggestions..")
        await asyncio.sleep(0.5)
        print("Wishing you a pleasant day!")

async def main():
    flower_chat = ChatOpenAI(
        max_tokens=100,
        streaming=True,
        callbacks=[FlowerSyncHandler(),FlowerAsyncHandler()]
    )

    await flower_chat.agenerate([[HumanMessage(content="Which flowers are best for birthday?Just mentions 3 types birefy,not exceeding 50 characters.")]])

asyncio.run(main())


"""
Constructing Token Counters with get_openai_callback
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
## Using ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback


llm = OpenAI(temperature=0.5,model_name="gpt-3.5-turbo-instruct")

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory())

with get_openai_callback() as cb:
    #round 1
    conversation("My sister's birthday is tomorrow,and i need a birthday bouquet.")
    # print("Memory after the first conversation:",conversation.memory.buffer)

    #round 2
    conversation("She likes pink roses,specifically the color pink.")
    # print("Memory after the second conversation:",conversation.memory.buffer)

    #round 3
    conversation("I'm back again.Do you remember why I came to buy flowers yseterday?")
    # print("Memory after the third conversation:",conversation.memory.buffer)


print("Total tokens used",cb.total_tokens)
#Total tokens used 494

async def interactions():
    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["What color flowers does my sister like?"]) for _ in range(3)]
        )
    print("Total tokens used in interactions",cb.total_tokens)

asyncio.run(interactions())
#Total tokens used in interactions 107

"""
hw:
Implement this token counter into other memory mechanisms?
"""

"""
Callbacks
Callbacks allow you to hook into the various stages of your LLM application's execution.

How to: pass in callbacks at runtime
How to: attach callbacks to a module
How to: pass callbacks into a module constructor
How to: create custom callback handlers
How to: use callbacks in async environments
How to: dispatch custom callback events
"""



