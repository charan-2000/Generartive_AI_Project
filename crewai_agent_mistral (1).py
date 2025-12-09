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

import pandas as pd
import sqlite3

# Load XLSX
df = pd.read_excel("insurance.xlsx")

# Create SQLite DB
conn = sqlite3.connect("insurance.db")
df.to_sql("insurance", conn, if_exists="replace", index=False)
conn.close()

print("Database created successfully!")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "/content/mistral_models/7B-Instruct-v0.3"  # change to your local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

mistral = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_sql(question):
    prompt = f"""
You are an expert in SQL.
The database table is named 'insurance' with columns:
age, sex, bmi, children, smoker, region, expenses.
Write ONLY the SQL query to answer:
"{question}"
    """
    output = mistral(prompt, max_length=1500, do_sample=False)[0]['generated_text']

    return output

def run_sql(query):
    conn = sqlite3.connect("insurance.db")
    result_df = pd.read_sql_query(query, conn)
    conn.close()
    return result_df

import re

def summarize_results(df, question):
    prompt = f"""
You are a data analyst. Here is the result of a SQL query answering:
"{question}"

{df.to_string(index=False)}

Write a clear and concise summary in natural language.
    """
    summary = mistral(prompt, max_length=2000, do_sample=False)[0]['generated_text']
    return summary

#print(sql_query)

user_question = "What is the average expenses of smokers from northwest region?"

sql_query = generate_sql(user_question)
print("Generated SQL:", sql_query)

sql_query_match = re.search(r"SELECT .*", sql_query, re.IGNORECASE | re.DOTALL)
if sql_query_match:
    sql_query = sql_query_match.group(0).strip()
else:
    raise ValueError("No SQL query found in sql_query.")

results = run_sql(sql_query)

print("SQL Results:\n", results)

final_summary = summarize_results(results, user_question)
print("\nSummary:\n", final_summary)

'''
import sqlite3

# Connect to your DB file
conn = sqlite3.connect("insurance.db")
cursor = conn.cursor()

# See tables in DB
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Preview records from a table
cursor.execute("SELECT * FROM insurance LIMIT 5;")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()

'''

"""#### Used MultiAgent with Mistral 7B"""

from crewai import Agent, Crew, Task
import sqlite3
import pandas as pd

df = pd.read_excel("insurance.xlsx")
conn = sqlite3.connect("insurance.db")

df.to_sql("insurance", conn, if_exists="replace", index=False )

#conn.close()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "/content/mistral_models/7B-Instruct-v0.3"  # change to your local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

mistral = pipeline("text-generation", model=model, tokenizer=tokenizer)

from crewai import LLM

mistral_llm = LLM(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    base_url="http://localhost:8000"
)

mistral_llm

#SQL_Reader- Agent1

sql_agent = Agent(
    role = "Text to SQL Converter",
    goal = "Take a question and convert it to valid sqlite sql query using insurance table",
    backstory = "You are a very good SQL converter with no explanation ",
    llm = mistral_llm
)

#DB executor
#from crewai_tools import BaseTool
from crewai.tools import BaseTool

def fetch_records(query):

  sql_query_match = re.search(r"SELECT .*", query, re.IGNORECASE | re.DOTALL)
  if sql_query_match:
      sql_query = sql_query_match.group(0).strip()
  else:
      raise ValueError("No SQL query found in sql_query.")
  result = pd.read_sql(sql_query, conn)


  return result.to_dict(orient = 'records')

class DBExecutorTool(BaseTool):
    name: str = "DB Executor"
    description: str  = "Executes SQL queries against the insurance database."

    def _run(self, sql_query: str) -> str:
        conn = sqlite3.connect("insurance.db")
        result = pd.read_sql_query(sql_query, conn)  # your SQLite connection
        conn.close()
        return result.to_json()  # return JSON string

db_tool = DBExecutorTool()

db_agent = Agent(
    role = "DB Executor",
    goal = "Execute sql query using sqlite database and return the results",
    backstory = "You have direct access to insurance databse you can run queries directly",
    tools=[db_tool],
    llm = mistral_llm
)

#summary agent
answer_agent = Agent(
    role = "Summary generator",
    goal = "the resuls given summarize it according to the given quetsion",
    backstory = "You are a expert data Analyst , summarise the results",
    llm = mistral_llm
)

question = "What is the average expenses of smokers from northwest region?"

task1 =  Task(agent=sql_agent, description=f"Write a sql query for the question: {question}", expected_output="A valid SQL query to answer the question with no explanation")
task2 =  Task(agent = db_agent, description="Execute the sql query and return results", expected_output="The results of the SQL query.")
task3 = Task(agent=answer_agent, description="Summarize the results", expected_output="A clear and concise summary of the SQL query results in natural human language.")

"""1️⃣ What CrewAI Actually Does
CrewAI is not a model itself — it’s an orchestrator for agents.
Each agent has:

A role (e.g., SQL Expert)

A goal (e.g., write optimized SQL)

A toolbox (could be search, database queries, or Python execution)

An LLM connection (how it thinks and responds)

When an agent needs to “think” or “respond,” CrewAI doesn’t do the AI work itself — instead, it calls an LLM API (like OpenAI, Anthropic, or your local model).

So, CrewAI must know:

Where the model lives (base_url)

Which model to use (model name or identifier)
"""

#!pip install pyngrok
from pyngrok import ngrok
# Replace 'YOUR_NGROK_AUTHTOKEN' with your actual authtoken
ngrok.set_auth_token("3190BvJkMWZmOdUIWAUVwjnWnqz_7xnwWh9S6v7MXQv9wV8jR")
from pyngrok import ngrok

# Open a tunnel to port 8000
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

crew = Crew(agents=[sql_agent, db_agent, answer_agent], tasks=[task1, task2, task3])
result = crew.kickoff()
print(result)

#Run Crew AI using FastAPI

!pip install fastapi uvicorn pyngrok transformers accelerate --quiet

from pyngrok import ngrok
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Body
import uvicorn
import threading

# ===== STEP 1: Load Model =====
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
print("Loading model... this may take a while ⏳")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Model loaded.")

# ===== STEP 2: Create FastAPI App =====
app = FastAPI()

@app.post("/generate")
def generate(prompt: str = Body(..., embed=True)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

# ===== STEP 3: Start ngrok =====
# Replace 'YOUR_NGROK_AUTHTOKEN' with your actual authtoken
ngrok.set_auth_token("3190BvJkMWZmOdUIWAUVwjnWnqz_7xnwWh9S6v7MXQv98jR")
public_url = ngrok.connect(8000)
print(f"🔗 Public URL: {public_url}")

# ===== STEP 4: Start FastAPI in background =====
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_server)
thread.start()

