#   Accessing models via HugggingFace hub
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HuggingFace API Token'


from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain

llm = HuggingFaceEndpoint(
    # repo_id="meta-llama/Llama-2-7b-chat-hf"
    # repo_id="google/flan-t5-small"   
    repo_id="google/gemma-7b"
)

template = """Question:{question}
                Answer:"""

prompt = PromptTemplate(template=template,input_variables=["question"])
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

question = "Rose is which type of flower?"

print(llm_chain.run(question))

#pipline