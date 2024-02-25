"""API routes for document management."""

import logging
from functools import lru_cache
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel

from vta_rag.storage.context import create_storage_context, get_storage_context

__all__ = ["router", "get_index"]

log = logging.getLogger(__name__)

embedder = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5", cache_folder="./models"
)

# reranker = LLMRerank(llm="BAAI/bge-large-en-v1.5", choice_batch_size=5, top_n=3)

# TODO: Tune this...
# - Buffer Size: Number of prev sentences to consider when deciding if current sentence is a new node.
# - Breakpoint Threshold: Cosine Similarity threshold before breaking
# NOTE: For some reason, unlike the SentenceSplitter, this doesn't have a length limit...
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

# TODO: Reranker, tune the node splitting strategy & retrieval strategy.


class CreateDocumentRequest(BaseModel):
    """Request model for creating a document."""

    doc_id: Optional[str]
    """ID for tracking like document owner and which dataset it belongs to."""
    ds_id: str
    """ID for the dataset the document belongs to."""
    content: str | List[str]
    """The content of the document. Specify list if pre-chunked."""
    overwrite: bool = False
    """Forcefully overwrite previous document is already exists. Else error."""


class DeleteDocumentRequest(BaseModel):
    """Request model for creating a document."""

    doc_id: str
    """ID for tracking like document owner and which dataset it belongs to."""
    ds_id: str
    """ID for the dataset the document belongs to."""


router = APIRouter()

# TODO: List documents
# TODO: Get chunk(s) by chunk id(s) so frontend can display them


# TODO: I alr CRUD the StorageContext, just need API routes to actually create/delete them.
# Each StorageContext represents a single dataset tied to a bot.
@router.on_event("startup")
def create_test_context():
    """Create a test context."""
    create_storage_context("00000000-0000-0000-0000-000000000000")


@lru_cache
def get_index(ds_id):
    """Get index from dataset id."""
    # db_ctx = get_storage_context(ds_id)
    # NOTE: This is temporary measure to make it more convenient downstream to use the API.
    db_ctx = create_storage_context(ds_id)
    # TODO: Figure out how to use ref_doc_id with external vector store & how to del documents.
    return VectorStoreIndex(
        [], service_context=srv_ctx, storage_context=db_ctx, store_nodes_override=True
    )


def del_doc(doc_id, ds_id):
    """Delete document from dataset."""
    index = get_index(ds_id)
    index.delete_ref_doc(doc_id, delete_from_docstore=True)
    get_storage_context(ds_id).persist()


# See how to CRUD documents from VectorStoreIndex:
# https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html


@router.post("/create_document/")
async def create_document(req: CreateDocumentRequest):
    """Create a new document."""
    text = req.content
    doc_id = req.doc_id or str(uuid4())
    ds_id = req.ds_id
    index = get_index(ds_id)

    # TODO: paraphrase document/chunks first?

    if isinstance(text, list):
        Document
    else:
        text = text.replace("\r\n", " ")
        text = text.replace("\n", " ")
        text = text.replace("\r", "")
        doc = Document(text=text, doc_id=doc_id)

    # TODO: THIS FAILS TO DETECT DOCUMENT ALR IN DATABASE WHEN USING WORKAROUND FOR VEC DBs
    if doc_id in index.ref_doc_info:
        if not req.overwrite:
            raise HTTPException(403, f"Document with id {doc_id} already exists.")
        del_doc(doc_id, ds_id)

    index.insert(doc)
    get_storage_context(ds_id).persist()
    node_ids = index.ref_doc_info[doc_id].node_ids
    return dict(doc_id=doc_id, ds_id=ds_id, node_ids=node_ids)


@router.post("/delete_document/")
async def delete_document(req: DeleteDocumentRequest):
    """Delete a document."""
    doc_id = req.doc_id or str(uuid4())
    ds_id = req.ds_id
    index = get_index(ds_id)
    node_ids = index.ref_doc_info[doc_id].node_ids

    del_doc(doc_id, ds_id)
    return dict(doc_id=doc_id, ds_id=ds_id, node_ids=node_ids)


# TODO: Return chunks by document id
# TODO: Check that the chunkNodes actually store the document id
