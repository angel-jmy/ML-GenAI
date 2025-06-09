#Hugging Pipline

from transformers import AutoTokenizer
import transformers
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1000
)

llm = HuggingFacePipeline(pipeline=pipline,
                          model_kwargs={'temperature':0})

template = """
                Generate a detailed and appealing description for the following bouquet:
                Bouquet Details:
                ```{flower_details}```
            """

prompt = PromptTemplate(template=template,
                        input_variables=["flower_details"])

llm_chain = LLMChain(prompt=prompt,llm=llm)

flower_details = "12 red roses, accompanied by white baby's breath and green leaves, wrapped in romantic red paper."

print(llm_chain.run(flower_details))