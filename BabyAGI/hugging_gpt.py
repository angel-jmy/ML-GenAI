from dotenv import load_dotenv
load_dotenv()

import os
from tavily import TavilyClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents.hugginggpt.hugginggpt import HuggingGPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

# === LLM Setup ===
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# === Tool 1: Weather Lookup via Tavily ===
@tool
def weather_analysis(city: str) -> str:
    """Fetches real-time weather info using Tavily for a given city."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Error] TAVILY_API_KEY not set."

    client = TavilyClient(api_key=api_key)
    results = client.search(query=f"Current weather in {city}", include_raw_content=True)
    if not results["results"]:
        return f"[Error] No results for {city}"
    
    best = results["results"][0]
    return f"{best['title']}\n{best['content']}\nSource: {best['url']}"

# === Tool 2: Flower Storage Strategy via GPT ===
flower_prompt = ChatPromptTemplate.from_template("""
You are an AI botanical storage expert.
Given the weather details below, output a flower storage strategy:

{weather}

Flower Storage Strategy:
""")
flower_chain = flower_prompt | llm | StrOutputParser()

@tool
def flower_strategy(weather: str) -> str:
    """Generates flower storage strategy based on weather text."""
    return flower_chain.invoke({"weather": weather})

# === Optional Memory (Not Used in HuggingGPT) ===
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["dummy"], embedding=embedding, metadatas=[{}], ids=["init"])
memory = vectorstore.as_retriever()

# === HuggingGPT Agent ===
agent = HuggingGPT(
    llm=llm,
    tools=[weather_analysis, flower_strategy],
)

# === Run Agent ===
objective = "Analyze the weather in San Francisco and create a flower storage strategy."
output = agent.run(objective)

print("\n===== Agent Output =====")
print(output)
