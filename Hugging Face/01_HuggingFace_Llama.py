from transformers import AutoTokenizer,AutoModelForCausalLM
import os
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HuggingFace API Token'
os.environ['HF_API_KEY'] = 'HuggingFace API Token'


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# # tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map = 'auto'
)

prompt = "translate English to German: How old are you?"

inputs = tokenizer(prompt,return_tensors="pt").to('cuda')

outputs = model.generate(inputs["input_ids"],max_new_tokens=2000)

response = tokenizer.decode(outputs[0],skip_special_tokens=True)

print(response)


# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))

