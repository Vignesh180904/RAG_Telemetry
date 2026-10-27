"""
LangChain retriever adapter
===========================
Concept: LangChain defines a standard `BaseRetriever` interface
(`.invoke(query) -> list[Document]`). By wrapping YOUR existing Retriever
in it, the agent graph (and any other LangChain/LangGraph code) can use your
ChromaDB search without knowing how it works -- and your original ingestion,
chunking and vector-store code stays untouched.
"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class ChromaRetrieverAdapter(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any            # app.retrieval.retriever.Retriever
    top_k: int = 3
    source_filter: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        chunks = self.retriever.retrieve(query, top_k=self.top_k, source_filter=self.source_filter)
        return [
            Document(
                page_content=c.text,
                metadata={"id": c.id, "source": c.source, "chunk_index": c.chunk_index, "score": c.score},
            )
            for c in chunks
        ]
