# from transformers import AutoTokenizer,AutoModelForCausalLM
# import os
# from transformers import T5ForConditionalGeneration


# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HuggingFace API Token'
# os.environ['HF_API_KEY'] = 'HuggingFace API Token'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model_name = "google/flan-t5-small"

# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# # tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     device_map = 'auto'
# # )

# model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)


# prompt = "translate English to German: How old are you?"

# # inputs = tokenizer(prompt,return_tensors="pt").to('cuda')
# inputs = tokenizer(prompt, return_tensors="pt")
# inputs = {k: v.to("cuda") for k, v in inputs.items()}


# # outputs = model.generate(inputs["input_ids"],max_new_tokens=2000)
# outputs = model.generate(inputs["input_ids"], max_new_tokens=50)


# response = tokenizer.decode(outputs[0],skip_special_tokens=True)

# print(response)


# # from transformers import T5Tokenizer, T5ForConditionalGeneration

# # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
# # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# # input_text = "translate English to German: How old are you?"
# # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # outputs = model.generate(input_ids)
# # print(tokenizer.decode(outputs[0]))


from transformers import MarianMTModel, MarianTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-de"  # English → German

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input sentence
english_text = "How old are you?"
inputs = tokenizer(english_text, return_tensors="pt").to(device)

# Translate
translated = model.generate(**inputs, max_new_tokens=50)
output_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(f"EN: {english_text}")
print(f"DE: {output_text}")
