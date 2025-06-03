"""
Chain

Agents:
Structured tool chat Agent\ self-Ask with search \ plan and Ececute Agent

What Structured tool?  LangChain v0.1 -> v0.3
File Management  Toolset
Web browser Toolset:PlayWright


PlayWright:Chrme Firfox safari...  pip install PlayWright->  PlayWright install

"""


from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()

        page = browser.new_page()
        page.goto('https://langchain.com/')

        title = page.title()
        print(f"Page title is:{title}")

        browser.close()

if __name__ == "__main__":
    run()
#Page title is:LangChain

# """
# Automation and Repetition
# Integration with CI/CD
# Cross-platform and Cross-browser testing
# logs report...

# """

#Structured tool chat Agent
# #AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION   ->PlayWrightBrowserTookit
# """
# https://python.langchain.com/v0.1/docs/integrations/toolkits/playwright/
# Some tools bundled within the PlayWright Browser toolkit include:

# NavigateTool (navigate_browser) - navigate to a URL
# NavigateBackTool (previous_page) - wait for an element to appear
# ClickTool (click_element) - click on an element (specified by selector)
# ExtractTextTool (extract_text) - use beautiful soup to extract text from the current web page
# ExtractHyperlinksTool (extract_hyperlinks) - use beautiful soup to extract hyperlinks from the current web page
# GetElementsTool (get_elements) - select elements by CSS selector
# CurrentPageTool (current_page) - get the current page URL
# """

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_community.tools.playwright.utils import create_async_playwright_browser
# # from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

# async_browser = create_async_playwright_browser()

# toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
# tools = toolkit.get_tools()

# print(tools)

# from langchain.agents import initialize_agent,AgentType
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(temperature=0.5)
# agent_chain = initialize_agent(tools,llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

# async def main():
#     response = await agent_chain.arun("What are the title on https://python.langchain.com")
#     print(response)

# import asyncio
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())



#2. self-Ask with search -SELF_ASK_WITH_SEARCH: FOLLOW-UP QUESTION+ Intermediate answer
#Muliti-hop question:
# """
# Tool set
# Stepwise Approach
# Self-questioning and Search
# Decision Chain
# """

# from langchain_openai import OpenAI
# from langchain_community.utilities import SerpAPIWrapper
# from langchain.agents import initialize_agent,Tool
# from langchain.agents import AgentType


# llm = OpenAI(temperature=0)
# search = SerpAPIWrapper()

# tools = [
#     Tool(
#         name="Intermediate Answer",
#         func=search.run,
#         description="useful for when you need to ask with search",
#     )
# ]

# self_ask_with_search = initialize_agent(tools,llm,agent=AgentType.SELF_ASK_WITH_SEARCH,verbose=True)

# self_ask_with_search.run("What is the capital of the country that uses the rose as its national flower?")#multi-hop

"""
> Entering new AgentExecutor chain...
 Yes.
Follow up: Which country uses the rose as its national flower?
Intermediate answer: Revered in poetry, film, theatre and music, it's quite understandable why the rose is the national flower of the United States, the United Kingdom and the Maldives.
Follow up: What is the capital of the United States?
Intermediate answer: Washington, D.C.
So the final answer is: Washington, D.C.

> Finished chain.
"""

#3. plan and Ececute Agent::pip install langchain-experimental
#Plan-and-Solve
"""
plan : LLM agent ->reasoning
Ececute: LLM ->calling tools
"""
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent,Tool
from langchain.agents import AgentType
from langchain.chains import LLMMathChain
from langchain_experimental.plan_and_execute import PlanAndExecute,load_agent_executor,load_chat_planner

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm,verbose=True)

search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer question about current events",
    ),
    Tool(
        name="Caculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer question about math",
    )
]

model = ChatOpenAI(temperature=0)

planner = load_chat_planner(model)
executor = load_agent_executor(model,tools,verbose=True)

agent = PlanAndExecute(planner=planner,executor=executor,verbose=True)
agent.invoke("How many bouquets of roses can $100 buy in New York?")

#steps=[Step(value='Determine the price of one bouquet of roses in New York.'), Step(value='Divide $100 by the price of one bouquet of roses to find out how many bouquets can be bought with $100.'), Step(value='Round down the result to the nearest whole number since you cannot buy a fraction of a bouquet.'), Step(value='Output the final number of bouquets that can be bought with $100.'), Step(value="Given the above steps taken, please respond to the user's original question. \n")]



"""
> Entering new PlanAndExecute chain...
steps=[Step(value='Determine the price of one bouquet of roses in New York.'), 
Step(value='Divide $100 by the price of one bouquet of roses to calculate how many bouquets can be bought.'), 
Step(value='Round down the result to the nearest whole number since you cannot buy a fraction of a bouquet.'), 
Step(value='Provide the final answer to the user.'), 
Step(value="Given the above steps taken, please respond to the user's original question. \n")]

> Entering new AgentExecutor chain...
Thought: I can use the search tool to find the current price of a bouquet of roses in New York.
Action:
```
{
  "action": "Search",
  "action_input": "current price of a bouquet of roses in New York"
}
```
Observation: ['Roses ; MF24-RRFLB. Red Roses Fancy Wrap Bouquet. $84.99 - $609.99 ; MF24-MIDNIGHTB. Blue Rose Bouquet - Midnight Blue Roses. $59.95 - $999.99 ;
 MF-RAINROSE.', "Send exquisite roses and flower delivery in New York.
  From the world's best producers, hand finished and presented in our signature gift box.
  ", 'Small and Petite Red Rose Bouquet ... $78 Arrangement size Standard Bouquet will be delivered approximately as pictured. ... 
  $114 Arrangement size Deluxe Additional ...', '101 roses ; $295.90. Hand-tied Bouquet (As Shown) ; $325.90. Basket ; $325.90. Box.', 
  '$125 Arrangement size Standard Bouquet will be delivered approximately as pictured. · Most Popular · $200 Arrangement size Premium We will add more blooms and ...',
   'Blush Garden Roses + Sydney Hale Candle. Regular Price: $136.00. Special Price $120.00. Shop Now. Items 1 to 48 of 203 total. PlantShed.com ...',
    "Our version of traditional one dozen roses that would truly brighten someone's day. Dozen Long Stem Roses (10) $135", 'Rose Bouquet · 
    $65 Arrangement size Standard 1 dozen roses · Most Popular. $130 Arrangement size Deluxe 2 dozen roses · $195 Arrangement size Premium 3 dozen roses.', 
    'At one Queens florist, where costs have soared in recent years, a bouquet is $72, up from $60 in 2019.', 
    '$132.95 Arrangement size Standard Bouquet will be delivered approximately as pictured. ... $152.95 Arrangement size Deluxe Additional flowers will be added to ...']Thought:

> Finished chain.
*****

Step: Determine the price of one bouquet of roses in New York.

Response:

> Entering new AgentExecutor chain...
Thought: I need to calculate the number of bouquets that can be bought with $100, but I first need to determine the price of one bouquet of roses in New York.

Action:
```
{
  "action": "Search",
  "action_input": "Price of one bouquet of roses in New York"
}
```
Observation: ['1-Dozen Red Rose Bouquet Just Roses. Same-Day Hand-Delivery! 1-Dozen Red Rose Bouquet. $69.00. Shop Now.', 
'$125 Arrangement size Standard Bouquet will be delivered approximately as pictured. ... 
$175 Arrangement size Deluxe Additional flowers will be added to enhance ...', 
'Roses ; MF24-RRFLB. Red Roses Fancy Wrap Bouquet. $84.99 - $609.99 ; MF24-MIDNIGHTB. Blue Rose Bouquet - Midnight Blue Roses. $59.95 - $999.99 ; MF-RAINROSE.', 'Fresh bouquet of beautiful blooms, selected and arranged in a glass vase by our expert florists Floral colors and varieties will vary depending on local availability The best possible value in each size available: 12 stems, 18 stems, 24 stems.', 'One Dozen Pink Roses - One Dozen Pink Roses , Arranged Beautifully in a Glass Vase. $250 Arrangement size Premium Three Dozen Pink Roses. $175 Arrangement size Deluxe Two 
Dozen Pink Roses.', 'CLASSIC ROSE ROYALE $150.00. Buy Now · Flames of Passion Dozen Roses. FLAMES OF PASSION $150.00. Buy Now · Peach & White Roses Bouquet. PEACH & WHITE ROSES ...', 'Roses ; Lovely Red Roses. $250.00 ; Pretty in Pink. $250.00 ; CONTEMPORARY HEART. $675.00 ; Red Roses. $150.00 ; Always on My Mind - Long Stemmed Red Roses. $85.00.', 'Be Happy Bouquet with Roses. $54.95 ; Be My Love Bouquet with Red Roses. $62.95 ; Always on My Mind - Long Stemmed Red Roses. $79.95 ; Make Me Blush - Dozen Long ...', 'THIS ITEM CANNOT BE PURCHASED ALONE. It can only be added to an existing order. Minimum order amount for delivery must be equal to or greater than $30.', "Always on My Mind - Long Stemmed Red Roses. $99.99 ; Thoughts of You Bouquet with Red Roses. $74.95 ; Love's Divine Bouquet - Long Stemmed Roses. $129.99."]
Thought:I found several prices for bouquets of roses in New York, ranging from $54.95 to $250.00.
 To calculate how many bouquets can be bought with $100, I will need to divide $100 by the price of one bouquet of roses.

Action:
```
{
  "action": "Final Answer",
  "action_input": "Approximately how many bouquets of roses can be bought with $100 in New York?"
}
```

> Finished chain.
*****

Step: Divide $100 by the price of one bouquet of roses to calculate how many bouquets can be bought.

Response: Approximately how many bouquets of roses can be bought with $100 in New York?

> Entering new AgentExecutor chain...
Thought: I need to calculate the number of bouquets that can be bought with $100 in New York and round down the result to the nearest whole number.

Action:
```
{
  "action": "Final Answer",
  "action_input": "You can buy 6 bouquets of roses with $100 in New York."
}
```

> Finished chain.
*****

Step: Round down the result to the nearest whole number since you cannot buy a fraction of a bouquet.

Response: You can buy 6 bouquets of roses with $100 in New York.

> Entering new AgentExecutor chain...
Thought: The final answer to provide to the user is the result of the calculation, which is "You can buy 6 bouquets of roses with $100 in New York."

Action:
```
{
  "action": "Final Answer",
  "action_input": "You can buy 6 bouquets of roses with $100 in New York."
}
```


> Finished chain.
*****

Step: Provide the final answer to the user.

Response: You can buy 6 bouquets of roses with $100 in New York.

> Entering new AgentExecutor chain...
Thought: The user's original question is about determining the price of one bouquet of roses in New York.
Action:
```
{
  "action": "Final Answer",
  "action_input": "The price of one bouquet of roses in New York is $16.67."
}
```

> Finished chain.
*****

Step: Given the above steps taken, please respond to the user's original question.


Response: The price of one bouquet of roses in New York is $16.67.
"""


"""
ReAct                               AgentType.ZERO_SHOT_REACT_DESCRIPTION
Structured tool chat                AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
self-Ask with search \              AgentType.SELF_ASK_WITH_SEARCH
plan and Ececute Agent              from langchain_experimental.plan_and_execute import PlanAndExecute

Converdational                        AgentType.CONVERSATIONAL_REACT_DESCRIPTION
openai functions                    AgentType.OPENAI_FUNCTIONS
openai Multi functions Agent ...... AgentType.OPENAI_MULTI_FUNCTIONS
"""
"""
Model->prompt->Chain ->memory->agents->indexs
"""

#hw: 
#The plan and Ececute Agent,please analyze the calling process of the PlanAndExcute,AgentExcutor amd LLMMathChain chains,as well as the agent's thought process.

