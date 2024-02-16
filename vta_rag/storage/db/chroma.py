"""Database handling for chromadb.

Only contains vector database for now.
"""

import logging

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from vta_rag.constants import TESTING_DIR
from vta_rag.storage.db.base import VectorDbCRUD

__all__ = ["ChromaDbCRUD"]

log = logging.getLogger(__name__)


def _wrap_chroma_db(table):
    """Wrap chroma collection into LlamaIndex."""
    vec_db = ChromaVectorStore(chroma_collection=table)
    return vec_db


class ChromaDbCRUD(VectorDbCRUD):
    """Chroma database dataset CRUD."""

    def __init__(self):
        """Init."""
        # NOTE: LlamaIndex handles the embedding, chromadb's built-in embedder is unused.
        self.db = chromadb.PersistentClient(str(TESTING_DIR / "chroma_db"))

    def create(self, ds_id):
        """Create new dataset."""
        try:
            table = self.db.create_collection(ds_id)
        except Exception as e:
            log.warning(
                "Collection already exists. "
                "In the future (after development), this error will be fatal."
            )
            log.warning(e)
            return self.get(ds_id)
        return _wrap_chroma_db(table)

    def get(self, ds_id):
        """Get existing dataset."""
        table = self.db.get_collection(ds_id)
        return _wrap_chroma_db(table)

    def delete(self, ds_id):
        """Delete dataset."""
        self.db.delete_collection(ds_id)
