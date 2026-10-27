"""
Prompt Construction
====================
Concept: We combine the user's question with the retrieved context into a
single instruction block for the LLM.

Why we need it: the LLM has no idea the retrieval step even happened --
all it sees is whatever text we hand it. The prompt is where we explicitly
tell it "only use this context" and "say you don't know if it's not here".
This is the single biggest lever we have against hallucination.
"""

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. Do not use any outside knowledge.\n"
    "If the answer is not present in the context, respond exactly with: "
    "\"I don't know based on the provided documents.\"\n"
    "Always be concise and cite which source number(s) you used."
)


def build_rag_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_INSTRUCTIONS}

Context:
{context}

Question: {query}

Answer:"""
