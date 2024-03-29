"""Create LlamaIndex storage context per dataset."""

from llama_index.core import StorageContext

from vta_rag.constants import LOCAL_TEST, TESTING_DIR

if LOCAL_TEST:
    from vta_rag.storage.db.local import LocalDbCRUD

    vec_db = LocalDbCRUD()
else:
    from vta_rag.storage.db.mongo import MongoDbCRUD

    vec_db = MongoDbCRUD()

__all__ = [
    "create_storage_context",
    "get_storage_context",
    "delete_storage_context",
]

# Of greatest concern to us is `docstore` & `vector_store`. `index_store` is useful for cache when scaling.
# https://docs.llamaindex.ai/en/stable/api_reference/storage.html
# See the full list of vector store sources supported:
# https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html


def create_storage_context(ds_id: str) -> StorageContext:
    """Create storage context for dataset."""
    # NOTE: Temporary.
    if not (TESTING_DIR / ds_id).exists():
        StorageContext.from_defaults().persist(str(TESTING_DIR / ds_id))

    # Note: If not set, it defaults to its "simple" variant, an in-memory store that can be dumped/loaded from disk.
    db_ctx = StorageContext.from_defaults(
        vector_store=vec_db.create(ds_id),
        persist_dir=str(TESTING_DIR / ds_id),
    )
    return db_ctx


def get_storage_context(ds_id: str) -> StorageContext:
    """Get storage context for dataset."""
    db_ctx = StorageContext.from_defaults(
        vector_store=vec_db.get(ds_id),
        persist_dir=str(TESTING_DIR / ds_id),
    )
    return db_ctx


def delete_storage_context(ds_id: str):
    """Delete storage context for dataset."""
    vec_db.delete(ds_id)
    # NOTE: I am not deleting during testing if I can help it, I don't need an rm -rf / incident.
    # (TESTING_DIR / ds_id).rmdir()
