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
        return "[Error] TAVILY_API_KEY not found in environment."

    try:
        client = TavilyClient(api_key=api_key)
        query = f"Current weather in {location}"
        result = client.search(query=query, include_raw_content=True)

        if not result or not result["results"]:
            return f"[Error] No results found for '{location}'."

        # Use first relevant result
        best = result["results"][0]
        title = best.get("title", "")
        snippet = best.get("content", "")
        url = best.get("url", "")

        return f"{title}\n{snippet}\nSource: {url}"

    except Exception as e:
        return f"[Error] Tavily search failed: {str(e)}"

# === Tool 2: Flower Storage Strategy via GPT ===
flower_prompt = ChatPromptTemplate.from_template("""
You are a botanical storage expert. Based on the current weather conditions below,
write a flower storage strategy that ensures optimal freshness and prevents damage.

Weather:
- Temperature: {temp_c}°C
- Humidity: {humidity}%
- Wind Speed: {wind_mph} mph
- Condition: {condition}

Strategy:
""")

flower_chain = flower_prompt | llm | StrOutputParser()

@tool
def flower_strategy(weather: str) -> str:
    """Generates flower storage strategy based on weather text."""
    import ast
    try:
        weather = ast.literal_eval(weather_info)
        strategy = flower_chain.invoke({
            "temp_c": weather.get("temp_c", "unknown"),
            "humidity": weather.get("humidity", "unknown"),
            "wind_mph": weather.get("wind_mph", "unknown"),
            "condition": weather.get("condition", "unknown")
        })
        print(strategy)
        return strategy

    except Exception as e:
        return f"[Error] Could not parse weather input: {e}"

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
