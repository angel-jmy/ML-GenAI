#LLM
#CAMEL: Can ChatGPT generate these guiding texts on its own?
"""
CAMEL: Interaction Agent Framework
CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society
CAMEL: Communication Agents Mind Exploration LLM

Communication Agents:

Inception Prompting: Task specifier prompt\ AI assistant prompts and AI user prompts

Task: Brainstorming a flower marketing scheme through role-playing.
1. Role-playing Agent
2. Task Specification
3. Initial prompt setup
4. Interaction norms
"""

from dotenv import load_dotenv
load_dotenv()

from typing import List
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

#1. Define the CAMEL Agent class
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
