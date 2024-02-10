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
    engine = index.as_query_engine(response_mode="no_text")
    # TODO: Investigate QueryBundle & Query Transformation.
    result = engine.query(req.query)
    nodes = result.source_nodes
    return dict(chunks=[(n.get_score(), n.get_text()) for n in nodes])
