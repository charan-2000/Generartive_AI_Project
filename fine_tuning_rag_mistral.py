# -*- coding: utf-8 -*-

import os
from pathlib import Path
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')


from huggingface_hub import snapshot_download

dir = Path.cwd()

print(dir)

mistral_models_path = dir.joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Retrieve the Hugging Face token from Colab secrets
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=mistral_models_path,
    token=hf_token # Pass the token to the snapshot_download function

)

#!pip install -r reuirements.txt

!python --version

!pip install mistral_inference mistral_common transformers streamlit torch transformers accelerate
#!pip install --upgrade torch transformers accelerate

#!pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flash-attn

from pathlib import  Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
dir = Path.cwd()


mistral_models_path = dir.joinpath('mistral_models', '7B-Instruct-v0.3')
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.8, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)

"""
🔍 1. AutoTokenizer – Converts Text ↔ Tokens
LLMs don’t read plain text like “hello world”.

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokens = tokenizer("Hello Mistral", return_tensors="pt") #A PyTorch tensor is just a multidimensional array — like a NumPy array, but made for deep learning.
print(tokens)

Tokenizer converts text into tokens (numbers) the model understands.SO to read the model from local need to import transformers lib

 2. AutoModelForCausalLM – Loads the Actual Brain
This is the real Mistral model you downloaded.

It processes the tokenized input and predicts the next words.

3. pipeline("text-generation") – Easy Text Completion
The pipeline() function connects everything.

It:

Uses your tokenizer.

Feeds tokens to the model.

Converts the output back to readable text."""

!pip install --upgrade --force-reinstall torchvision

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
model_path  = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",device_map="auto")
model.gradient_checkpointing_enable()

'''
torch_dtype="auto"
Tells the model to automatically choose the best precision (float16, bfloat16, or float32) based on your GPU/CPU.

Benefit: Faster and uses less memory on supported hardware (e.g., T4, A100, etc.)

. device_map="auto"
Automatically distributes the model across available devices:

If you have one GPU, it loads the model there.

If you're on CPU, it l  oads to CPU.

If multiple GPUs, it splits layers across them.
'''

#!pip install transformers -U

question = 'Tell me about Abdul Kalam?'

prompt = f"<s> <INST> {question} </INST>"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

resp = model.generate(**inputs, temperature = 0.3, max_new_tokens=100, do_sample=True, top_p=0.9)

#print(resp)

output = tokenizer.decode(resp[0], skip_special_tokens=True)

print(output)

question = 'Tell me about Virat Kohli biography and his achievements in IPL?'

#question = 'Tell me about Virat Kohli biography?'
prompt = f"<s> <INST> {question} </INST>"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

resp = model.generate(**inputs, temperature = 0.3, max_new_tokens=1000, do_sample=True, top_p=0.9)

#print(resp)

output = tokenizer.decode(resp[0], skip_special_tokens=True)

print(output)

import streamlit as st

st.title("Mistral LLM Q&A Interface ")

input_text = st.text_input("Serach for the topic")


prompt = f"<s> <INST> {input_text} </INST>"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

resp = model.generate(**inputs, temperature = 0.3, max_new_tokens=100, do_sample=True, top_p=0.9)

#print(resp)

output = tokenizer.decode(resp[0], skip_special_tokens=True)

#print(output)

if input_text:
    st.write(output)

#!streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py

"""#### FINE TUNING USINF PEFT"""

!pip install -q datasets accelerate evaluate trl accelerate bitsandbytes peft

import torch
print(torch.cuda.get_device_name(0))  # Should print your 40GB card

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ✅ Fix: Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load Excel and preprocess
df = pd.read_excel("insurance.xlsx")

# Convert to prompt-completion format
df["text"] = df.apply(lambda row:
    f"Input: Age={row['age']}, Sex={row['sex']}, BMI={row['bmi']}, Children={row['children']},Smoker={row['smoker']}, Region={row['region']}\nOutput: Expenses={row['expenses']}", axis=1)

df = df.dropna(subset=["text"])  # Ensure no NaNs
dataset = Dataset.from_pandas(df[["text"]])
dataset = dataset.train_test_split(test_size=0.1)


# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds = tokenized_ds.remove_columns(["text"])

# Apply LoRA using PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral7b-lora-insurance",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    #evaluation_strategy="steps",
    logging_steps=20,
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-4,
    num_train_epochs=2,
    warmup_steps=10,
    bf16=True,
    gradient_accumulation_steps=2,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=data_collator
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./mistral7b-lora-insurance")
tokenizer.save_pretrained("./mistral7b-lora-insurance")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # if needed

# Load LoRA fine-tuned weights
model = PeftModel.from_pretrained(base_model, "mistral7b-lora-insurance")
model.eval()

# 🧠 Now you're ready to ask questions
prompt = "What is the best way to handle customer complaints?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""### Example-1"""

question = "What is the average monthly expense for a 22-year-old?"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""- In the above it gave a generic answer but i want related to my trained datset. so i will be giving this as a  try"""

#question = "use the training data only" + "What is the average monthly expense for a 22-year-old?"
question = "what is the average expense of smokers in each region"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""#### RAG (RETRIVAL AUGMENTED GENERATION)"""

import pandas as pd

#Turn your structured data into readable text chunks.

# Load Excel
df = pd.read_excel("insurance.xlsx")

# Convert rows to text
chunks = []
for _, row in df.iterrows():
    text = (
        f"Age: {row['age']}, Sex: {row['sex']}, BMI: {row['bmi']}, "
        f"Children: {row['children']}, Smoker: {row['smoker']}, "
        f"Region: {row['region']}, Expenses: ${row['expenses']}"
    )
    chunks.append(text)

chunks[2]

"""Step 2: Vectorize Chunks <br>
Use an embedding model to convert chunks to vectors.
"""

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can replace this with a local model if needed
vectors = embed_model.encode(chunks, convert_to_tensor=True)

vectors

#!pip install faiss-cpu

"""- Use FAISS for fast similarity search."""

import faiss #Facebook AI Similarity Search
import numpy as np

# Convert to float32 for FAISS
embeddings = np.array(vectors.cpu()).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

index

"""- Step 4: Accept a User Query <br>
Convert query to vector and search for most relevant chunks.
"""

def get_top_chunks(query, k=10):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

"""- Step 5: Use LLM to Answer <br>
Pass the retrieved text into an LLM for final answer.
"""

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

'''
response = generator(
    prompt,
    max_new_tokens=256,  # Avoid using max_length
    do_sample=False,
    truncation=True,     # Add this to remove warning
    pad_token_id=tokenizer.eos_token_id  # Explicit pad token
)[0]['generated_text']
'''
query = "What is the average expense for smokers in the northwest?"
top_chunks = get_top_chunks(query)
context = "\n".join(top_chunks)

prompt = f"Based on the data below, answer the following question:\n{query}\n\n{context}\n\nAnswer:"
response = generator(prompt,truncation=True, max_length=300, do_sample=False)[0]['generated_text']
print(response)

