from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set up Azure OpenAI environment variables
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://langchain12.openai.azure.com/"  # Base URL of the Azure endpoint
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template
prompt_template = """You are a helpful assistant. Respond to user queries.
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Streamlit app
st.title("LangChain Demo")
input_text = st.text_input("Search the topic you want")

# Azure OpenAI LLM setup
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-4",
    api_version="2024-08-01-preview",
    deployment_name="gpt-4",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

output_parser = StrOutputParser()

# Creating the chain
chain = prompt | llm | output_parser

# Run the chain if input is provided
if input_text:
    result = chain.invoke({"question": input_text})
    st.write(result)
