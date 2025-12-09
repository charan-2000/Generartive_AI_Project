#!/usr/bin/env python
# coding: utf-8

# #### Used MultiAgent with Mistral 7B

# 1️⃣ What CrewAI Actually Does
# CrewAI is not a model itself — it’s an orchestrator for agents.
# Each agent has:
# 
# A role (e.g., SQL Expert)
# 
# A goal (e.g., write optimized SQL)
# 
# A toolbox (could be search, database queries, or Python execution)
# 
# An LLM connection (how it thinks and responds)
# 
# When an agent needs to “think” or “respond,” CrewAI doesn’t do the AI work itself — instead, it calls an LLM API (like OpenAI, Anthropic, or your local model).
# 
# So, CrewAI must know:
# 
# Where the model lives (base_url)
# 
# Which model to use (model name or identifier)
