"""Main entrypoint."""

from fastapi import FastAPI
from pydantic import BaseModel

from vta_rag.document import get_index
from vta_rag.document import router as doc_router

app = FastAPI()
app.include_router(doc_router, prefix="/api/v1")


class QueryRequest(BaseModel):
    """Request model for creating a document."""

    ds_id: str
    """ID for the dataset to query within."""
    query: str
    """The query."""


@app.post("/api/v1/query")
async def query(req: QueryRequest):
    """Query dataset for chunks."""
    index = get_index(req.ds_id)
    retriever = index.as_retriever(similarity_top_k=15)
    # TODO: Investigate QueryBundle & Query Transformation.
    result = await retriever.aretrieve(req.query)
    # reranker = LLMRerank(choice_batch_size=5, top_n=3)
    nodes = result  # reranker.postprocess_nodes(result, query_str=req.query)
    return dict(chunks=[(n.get_score(), n.get_text()) for n in nodes])
