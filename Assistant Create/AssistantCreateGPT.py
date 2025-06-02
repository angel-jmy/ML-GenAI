"""
Assistant

An business case using LLM.

🔍 Analyzing selling Songs data generate industry report and create PPT.
Objective: pop music trend analysis.
Project workflow:
Creating assistant and conversation -> AI-generated data visualizations -> data insights -> LLM-generated PPT titles and images -> integrating content to create the final PPT.
"""

# Creating assistant and conversation

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

import pandas as pd
file_path = "./Spotify_Songs.csv"
sales_data = pd.read_csv(file_path, nrows=20)
sales_data.head()

# from langchain_openai import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.agents import create_openai_functions_agent, tool

# class FileTool:
#     def __init__(self, file_path):
#         self.file_path = file_path

#     def run(self):
#         with open(self.file_path, 'rb') as file:
#             return file.read().decode()

# @tool
# def process_data(file_content: str):
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from io import StringIO
#
#     data = pd.read_csv(StringIO(file_content))
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['column1'], data['column2'])
#     plt.title('Column1 vs Column2')
#     plt.xlabel('Column1')
#     plt.ylabel('Column2')
#     plt.savefig('visualization.png')
#
#     return "Data processed and visualization created."

# template = """
# Given the data and a query, you can write appropriate code and create appropriate visualizations.
# Data: {data}
# """
#
# llm = OpenAI(model="gpt-4-0125-preview")
# prompt = PromptTemplate(template=template, input_variables=["data"])
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# agent_executor = create_openai_functions_agent(llm_chain, tools=[process_data])
#
# file_path = 'path_to_your_file.csv'
# file_path = 'Spotify_Songs.csv'
# file_tool = FileTool(file_path)
# file_content = file_tool.run()
# response = agent_executor({"data": file_content})
#
# print(response)







