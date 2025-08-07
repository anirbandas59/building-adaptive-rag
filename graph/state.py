"""
The foundation of the adaptive RAG system - the state management
It defines how information flows through our graph. It acts as the
central data structure that flows through every node in our graph workflow.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attributes:
        question: Question or user's input query
        generation: stores the LLM generation
        web_search: Whether to search web
        documents: List of all retrieved documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
