#What is a Large Language Model?-mass data---AGI---ChatGPT
"""
AIGC

DALL-E Stable Diffusion
AudioLM

Narrow AI->AGI: AI Killer APP(ChatGPT)

The Four phases of AI-induced"
1. Google Microsoft Facebook OpenAI Mistral AI
2. AI infrastructure
3. Cloudflare Autodesk MongoDB
4. AI automation

Natural Language programming  Paradigm: prompt engineering:
- Instructional
- Completion
- Scenario-based
- Demonstrative-COT

Chat model

Text Model

1. API Key
2. pip install openai
3. OpenAI API Key

export OPENAI_API_KEY="your_api_key_here"

# import os
# os.environ["OPENAI_API_KEY"] = "OPENAI API Key"

# import openai
# openai.api_key = "OPENAI API Key"
"""

from dotenv import load_dotenv
load_dotenv()  # .env

from openai import OpenAI
client = OpenAI()

# Text Model - API GPT-3

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    temperature=0.9, # Default is 0.7 (0 is deterministic)
    prompt="Write a tagline for an ice cream shop."
)
print(response.choices[0].text.strip())
# print(response)

"""
Completion(
  id='cmpl-AASyEDfZcwXUx38CbKscIjecpZ4AY',
  choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text='\n\n"Scoops of happiness in every bite!"')],
  created=1727858006,
  model='gpt-3.5-turbo-instruct',
  object='text_completion',
  system_fingerprint=None,
  usage=CompletionUsage(completion_tokens=10, prompt_tokens=10, total_tokens=20)
)
"""

# Chat Model - GPT-4o
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.9, # Default is 0.7 (0 is deterministic)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

print(response.choices[0].message.content)
# print(response)

# The 2020 World Series was played at Globe Life Field in Arlington, Texas. This was the first time the World Series was held at a neutral site due to the COVID-19 pandemic.

# The 2020 World Series was played at Globe Life Field in Arlington, Texas. This was notable because it was the first time the World Series was held at a neutral site due to the COVID-19 pandemic.
