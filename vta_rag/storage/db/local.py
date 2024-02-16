"""Database handling for local testing.

Only contains vector database for now.
"""

import logging

from llama_index.core.vector_stores.simple import (
    DEFAULT_PERSIST_FNAME,
    SimpleVectorStore,
)

from vta_rag.constants import TESTING_DIR
from vta_rag.storage.db.base import VectorDbCRUD

__all__ = ["LocalDbCRUD"]

log = logging.getLogger(__name__)


class LocalDbCRUD(VectorDbCRUD):
    """Local database dataset CRUD."""

    def __init__(self):
        """Init."""

    def create(self, ds_id):
        """Create new dataset."""
        vec_db = SimpleVectorStore()
        if (TESTING_DIR / ds_id / DEFAULT_PERSIST_FNAME).exists():
            log.warning("Dataset already exists.")
            return self.get(ds_id)
        vec_db.persist(str(TESTING_DIR / ds_id / DEFAULT_PERSIST_FNAME))
        return vec_db

    def get(self, ds_id):
        """Get existing dataset."""
        vec_db = SimpleVectorStore.from_persist_dir(str(TESTING_DIR / ds_id))
        return vec_db

    def delete(self, ds_id):
        """Delete dataset."""
        # NOTE: I am not deleting during testing if I can help it, I don't need an rm -rf / incident.
        # (TESTING_DIR / ds_id).rmdir()
