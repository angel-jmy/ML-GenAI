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





