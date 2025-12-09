#!/usr/bin/env python
# coding: utf-8

# 
# 🔍 1. AutoTokenizer – Converts Text ↔ Tokens
# LLMs don’t read plain text like “hello world”.
# 
# from transformers import AutoTokenizer
# 
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# tokens = tokenizer("Hello Mistral", return_tensors="pt") #A PyTorch tensor is just a multidimensional array — like a NumPy array, but made for deep learning.
# print(tokens)
# 
# Tokenizer converts text into tokens (numbers) the model understands.SO to read the model from local need to import transformers lib
# 
#  2. AutoModelForCausalLM – Loads the Actual Brain
# This is the real Mistral model you downloaded.
# 
# It processes the tokenized input and predicts the next words.
# 
# 3. pipeline("text-generation") – Easy Text Completion
# The pipeline() function connects everything.
# 
# It:
# 
# Uses your tokenizer.
# 
# Feeds tokens to the model.
# 
# Converts the output back to readable text.

# #### FINE TUNING USINF PEFT

# ### Example-1

# - In the above it gave a generic answer but i want related to my trained datset. so i will be giving this as a  try

# #### RAG (RETRIVAL AUGMENTED GENERATION)

# Step 2: Vectorize Chunks <br>
# Use an embedding model to convert chunks to vectors.

# - Use FAISS for fast similarity search.

# - Step 4: Accept a User Query <br>
# Convert query to vector and search for most relevant chunks.

# - Step 5: Use LLM to Answer <br>
# Pass the retrieved text into an LLM for final answer.
