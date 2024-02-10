"""Database handling.

Only contains vector database for now.
"""

import logging

import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore

from vta_rag.constants import TESTING_DIR

__all__ = [
    "create_vector_store",
    "get_vector_store",
    "delete_vector_store",
]

log = logging.getLogger(__name__)

# NOTE: LlamaIndex handles the embedding, chromadb's built-in embedder is unused.

db = chromadb.PersistentClient(str(TESTING_DIR / "chroma_db"))


def _wrap_chroma_db(table):
    """Wrap chroma collection into LlamaIndex."""
    vec_db = ChromaVectorStore(chroma_collection=table)
    return vec_db


def create_vector_store(ds_id: str) -> VectorStore:
    """Create new dataset."""
    try:
        table = db.create_collection(ds_id)
    except Exception as e:
        log.warning(
            "Collection already exists."
            "In the future (after development), this error will be fatal."
        )
        log.warning(e)
        return get_vector_store(ds_id)
    return _wrap_chroma_db(table)


def get_vector_store(ds_id: str) -> VectorStore:
    """Get existing dataset."""
    table = db.get_collection(ds_id)
    return _wrap_chroma_db(table)


def delete_vector_store(ds_id: str):
    """Delete dataset."""
    db.delete_collection(ds_id)
