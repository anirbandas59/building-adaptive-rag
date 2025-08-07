"""
This document grading implements quality control mechanism by evaluating
each retrieved document for relevance to the user's question. It iterates through
all retrieved documents and uses our retrieval grader to assess their relevance.
"""

from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered our irrelevant documents
    """
    print("---- CHECK DOCUMENT RELEVANCE TO QUESTION ----")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score

        if grade.lower() == "yes":
            print("---- GRADE: DOCUMENT RELEVANT ----")
            filtered_docs.append(d)
        else:
            print("---- GRADE: DOCUMENT NOT RELEVANT ----")
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
