#   Accessing models via HugggingFace hub
# import os
# os.environ['HF_API_KEY'] = 'HuggingFace API Token'


# from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
# # from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import LLMChain

# # llm = HuggingFaceEndpoint(
# #     # repo_id="meta-llama/Llama-2-7b-chat-hf"
# #     # repo_id="google/flan-t5-small"   
# #     repo_id="google/gemma-7b"
# # )


# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-small",
#     model_kwargs={"temperature": 0.5, "max_length": 100}
# )

# template = """Question:{question}
#                 Answer:"""

# prompt = PromptTemplate(template=template,input_variables=["question"])
# llm_chain = LLMChain(
#     prompt=prompt,
#     llm=llm
# )

# question = "Rose is which type of flower?"

# print(llm_chain.run(question))

#pipline


# from dotenv import load_dotenv
# import os
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_huggingface import HuggingFaceEndpoint  # ✅ new package
# from huggingface_hub import InferenceClient

# # Load token from .env
# load_dotenv()
# token = os.getenv("HF_API_KEY")

# # Setup client for a T5 model (text2text-generation)
# client = InferenceClient(model="google/flan-t5-small", token=token)

# # Define prompt
# prompt = "What is the capital of France?"

# # Call text2text-generation endpoint
# response = client.text2text_generation(prompt)
# print(response)


from transformers import pipeline

# Load a general-purpose text generation model
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)

# Ask your question
question = "What is the capital of France?"

# Generate an answer
output = generator(question, max_new_tokens=50, do_sample=False)

# Print the result
print("Answer:", output[0]['generated_text'])

