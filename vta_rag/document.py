"""API routes for document management."""

from fastapi import APIRouter
from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel

from vta_rag.storage.context import create_storage_context, get_storage_context

__all__ = ["router", "get_index"]

embedder = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5", cache_folder="./models"
)

# TODO: Tune this...
# - Buffer Size: Number of prev sentences to consider when deciding if current sentence is a new node.
# - Breakpoint Threshold: Cosine Similarity threshold before breaking
# NOTE: For some reason, unlikek the SentenceSplitter, this doesn't have a length limit...
splitter = SemanticSplitterNodeParser(
    buffer_size=2, breakpoint_percentile_threshold=90, embed_model=embedder
)

# TODO: ServiceContext is how you customize node splitting, embedding & retrieval
# strategy among many other things. Unfortunately, we aren't using most of those
# many other things lol.
# nvm its deprecated now lmao use a global singleton instead, forget about per user configs.
# I bet ServiceContext should be tied to user_id for different tiers.
# For now we use their defaults for node splitting & embedding.
# https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context.html
srv_ctx = ServiceContext.from_defaults(
    llm=None, embed_model=embedder, node_parser=splitter
)


# TODO: I alr CRUD the StorageContext, just need API routes to actually create/delete them.
# Each StorageContext represents a single dataset tied to a bot.
create_storage_context("00000000-0000-0000-0000-000000000000")

# TODO: Reranker, tune the node splitting strategy & retrieval strategy.


class CreateDocumentRequest(BaseModel):
    """Request model for creating a document."""

    doc_id: str
    """ID for tracking like document owner and which dataset it belongs to."""
    ds_id: str
    """ID for the dataset the document belongs to."""
    content: str
    """The content of the document."""


router = APIRouter()


def get_index(ds_id):
    """Get index from dataset id."""
    db_ctx = get_storage_context(ds_id)
    index = VectorStoreIndex(
        [],
        service_context=srv_ctx,
        storage_context=db_ctx,
        use_async=False,
        show_progress=True,
        store_nodes_override=True,
    )
    return index


# See how to CRUD documents from VectorStoreIndex:
# https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html


@router.post("/create_document/")
async def create_document(req: CreateDocumentRequest):
    """Create a new document."""
    text = req.content
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", "")
    doc = Document(text=text, doc_id=req.doc_id)
    index = get_index(req.ds_id)
    # VectorStoreIndex adds the nodes into the storage_context.docstore.
    # TODO: Isn't this duplicating what is stored by the vector store? Investigate.
    index.insert(doc)
    # TODO: Delete/overwrite document and past nodes if it already exists.
    get_storage_context(req.ds_id).persist()
    return len(index.ref_doc_info[doc.doc_id].node_ids)


# TODO: Return chunks by document id