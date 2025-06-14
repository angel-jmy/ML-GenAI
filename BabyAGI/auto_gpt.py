from dotenv import load_dotenv
load_dotenv()

# === LangChain imports ===
from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain_community.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import os
# from serpapi import GoogleSearch
from tavily import TavilyClient

# === Set up LLM ===
llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# === Define Tools ===
# def weather_analysis(task: str) -> str:
#     return f"[Simulated] Analyzing weather for task: {task}"

def weather_analysis(location: str) -> str:
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


# print(weather_analysis("San Francisco"))


# Define the prompt template
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

# Build the LLM chain
flower_chain = flower_prompt | llm | StrOutputParser()

# Define the tool function
def flower_strategy(weather_info: str) -> str:
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


tools = [
    Tool(
        name="WeatherAnalysisTool",
        func=weather_analysis,
        description="Uses TAVILY to search for real-time weather information by location."
    ),
    # Tool(
    #     name="WeatherAnalysisTool",
    #     func=weather_analysis,
    #     description="Analyzes the weather for a given location or context."
    # ),
    Tool(
        name="FlowerStrategyTool",
        func=flower_strategy,
        description="Generates a flower storage strategy based on weather or other input."
    )
]



# === Set up FAISS Retriever as Memory ===
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    texts=["Make a todo list"],  # Prevent empty index
    embedding=embedding,
    metadatas=[{"task": "Make a todo list"}],
    ids=["result_0_0"]
)
memory = vectorstore.as_retriever()  

# === Instantiate AutoGPT Agent ===
auto_gpt = AutoGPT.from_llm_and_tools(
    ai_name="WeatherFlowerGPT",
    ai_role="An autonomous agent that analyzes weather and writes flower storage strategies.",
    tools=tools,
    llm=llm,
    memory=memory,
)

# === Step 5: Run Agent with Objective ===
objective = "Analyze the weather in San Francisco today and write a flower storage strategy."
# auto_gpt.run(objective)
final_result = auto_gpt.run(objective)
# print("\n[FINAL RESULT]")
# print(final_result)


"""
OUTPUTS:

{
    "thoughts": {
        "text": "I need to analyze the weather in San Francisco today and write a flower storage strategy. To do this, I will first use the WeatherAnalysisTool to get the current weather in San Francisco. After that, I will use the FlowerStrategyTool to generate a flower storage strategy based on the weather data.",
        "reasoning": "The first step in creating a flower storage strategy is to understand the current weather conditions, as they can significantly impact the health and longevity of the flowers. Once I have this information, I can use it to inform the flower storage strategy.",
        "plan": "- Use WeatherAnalysisTool to get weather in San Francisco\n- Use FlowerStrategyTool to generate flower storage strategy",
        "criticism": "I need to ensure that I accurately interpret the weather data and apply it correctly to the flower storage strategy. Misinterpretation could lead to an ineffective strategy.",
        "speak": "I'm going to start by analyzing the weather in San Francisco using the WeatherAnalysisTool. Then, I'll use the FlowerStrategyTool to create a flower storage strategy based on the weather conditions."
    },
    "command": {
        "name": "WeatherAnalysisTool",
        "args": {
            "tool_input": "San Francisco"
        }
    }
}
{
    "thoughts": {
        "text": "Now that I have the weather data for San Francisco, I can use this information to generate a flower storage strategy. The weather data includes temperature, humidity, wind speed, and other factors that can affect the health of flowers. I will use the FlowerStrategyTool to generate a strategy based on this data.",
        "reasoning": "The weather conditions can significantly impact the health and longevity of flowers. For example, high temperatures can cause flowers to wilt, while high humidity can promote the growth of mold. Therefore, it's important to consider these factors when creating a flower storage strategy.",
        "plan": "- Use FlowerStrategyTool to generate flower storage strategy based on the weather data",
        "criticism": "I need to ensure that I accurately interpret the weather data and apply it correctly to the flower storage strategy. Misinterpretation could lead to an ineffective strategy.",
        "speak": "Now that I have the weather data, I'm going to use the FlowerStrategyTool to create a flower storage strategy based on these conditions."
    },
    "command": {
        "name": "FlowerStrategyTool",
        "args": {
            "tool_input": "{'temp_c': 17.2, 'humidity': 63, 'wind_kph': 20.9, 'condition': {'text': 'Partly cloudy', 'code': 1003}, 'precip_mm': 0.0}"
        }
    }
}
1. Temperature Control: The current temperature of 17.2°C is within the optimal range for most flowers. However, some tropical flowers may require slightly warmer conditions. Ensure that the storage facility has a temperature control system to maintain this temperature consistently.

2. Humidity Control: The current humidity level is 63%, which is slightly higher than the ideal range for most flowers (40-60%). Therefore, use a dehumidifier to reduce the humidity level in the storage area. This will prevent excess moisture from causing mold or rot.

3. Light Control: As the weather condition is partly cloudy, there might not be enough natural light for the flowers. Ensure that the storage facility has adequate artificial lighting to compensate for the lack of natural light. However, avoid direct sunlight as it can cause the flowers to wilt.

4. Wind Speed: The wind speed is unknown, but it's important to note that strong winds can cause physical damage to flowers. Ensure that the storage facility is well-protected from wind, regardless of its speed.

5. Regular Monitoring: Regularly monitor the temperature, humidity, and light levels in the storage facility to ensure they remain within the optimal range for flower storage.

6. Proper Storage: Store flowers in clean containers and change the water regularly to prevent bacterial growth. Also, remove any wilted or damaged flowers promptly to prevent them from affecting the others.

7. Ventilation: Ensure the storage area is well-ventilated to prevent the build-up of ethylene gas, which can cause flowers to age faster.

8. Hydration: Keep the flowers well-hydrated, but avoid overwatering as this can lead to root rot.

By following these strategies, we can ensure the flowers remain fresh and undamaged despite the current weather conditions.
{
    "thoughts": {
        "text": "I have successfully generated a flower storage strategy based on the current weather conditions in San Francisco. The strategy includes temperature control, humidity control, light control, wind speed considerations, regular monitoring, proper storage, ventilation, and hydration. There are no further commands to execute at this point.",
        "reasoning": "The generated flower storage strategy is comprehensive and takes into account all the necessary factors that can affect the health and longevity of flowers. Therefore, there is no need for further commands.",
        "plan": "- Finish the task",
        "criticism": "I have successfully completed the task. However, I should always remember to review the strategy to ensure it is practical and feasible.",
        "speak": "I have created a comprehensive flower storage strategy based on the current weather conditions in San Francisco. The strategy includes considerations for temperature, humidity, light, wind speed, regular monitoring, proper storage, ventilation, and hydration."
    },
    "command": {
        "name": "finish",
        "args": {
            "response": "I have successfully completed the task of creating a flower storage strategy based on the current weather conditions in San Francisco."
        }
    }
}



"""