"""
This step fetches relevant documents from the local vectorstore. It takes the user's
question from the state and uses our pre-configured retriever to find the most semantically
similar documents.
"""

from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---- RETRIEVE ----")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
