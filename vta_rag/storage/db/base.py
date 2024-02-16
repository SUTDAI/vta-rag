"""Base types for database."""

from llama_index.core.vector_stores.types import VectorStore

__all__ = ["VectorDbCRUD"]


class VectorDbCRUD:
    """Base class for vector database CRUD operations."""

    def __init__(self):
        """Init."""
        pass

    def create(self, ds_id: str) -> VectorStore:
        """Create new dataset using dataset id."""
        raise NotImplementedError

    def get(self, ds_id: str) -> VectorStore:
        """Get existing dataset using dataset id."""
        raise NotImplementedError

    def delete(self, ds_id: str):
        """Delete dataset using dataset id."""
        raise NotImplementedError
