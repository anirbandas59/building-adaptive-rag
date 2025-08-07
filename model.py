"""
Define chat model & the embedding model
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

llm_model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
