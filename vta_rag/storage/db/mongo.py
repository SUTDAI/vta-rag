"""Database handling for mongodb.

Only contains vector database for now.
"""

import logging

import pymongo
import requests
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from requests.auth import HTTPDigestAuth

from vta_rag.constants import (
    MONGO_ADMIN_KEY,
    MONGO_ADMIN_USER,
    MONGO_CLUSTER_NAME,
    MONGO_EP,
    MONGO_GROUP_ID,
    MONGO_URI,
    MONGO_VEC_DB_NAME,
)
from vta_rag.storage.db.base import VectorDbCRUD

__all__ = ["MongoDbCRUD"]

log = logging.getLogger(__name__)


def _wrap_mongo_db(mongo_client, ds_id: str):
    """Wrap chroma collection into LlamaIndex."""
    vec_db = MongoDBAtlasVectorSearch(
        mongo_client,
        db_name="vta-rag-vec-db",
        collection_name=ds_id,
        index_name="index",
    )
    return vec_db


class MongoDbCRUD(VectorDbCRUD):
    """Mongo database dataset CRUD."""

    def __init__(self):
        """Init."""
        self.db = pymongo.MongoClient(MONGO_URI)

    def create(self, ds_id: str) -> VectorStore:
        """Create new dataset."""
        res = requests.post(
            f"{MONGO_EP}/groups/{MONGO_GROUP_ID}/clusters/{MONGO_CLUSTER_NAME}/fts/indexes",
            auth=HTTPDigestAuth(MONGO_ADMIN_USER, MONGO_ADMIN_KEY),
            headers=dict(Accept="application/vnd.atlas.2023-01-01+json"),
            json=dict(
                collectionName=ds_id,
                database=MONGO_VEC_DB_NAME,
                name="index",
                type="vectorSearch",
                fields=[
                    dict(
                        type="vector",
                        path="embedding",
                        numDimensions=1024,
                        similarity="dotProduct",
                    )
                ],
            ),
        )

        if res.status_code != 200:
            log.warning(f"Error {res.status_code} while creating index for {ds_id}.")
            log.warning(res.json())
        else:
            log.info(f"Index for {ds_id} created.")
            log.info(res.json())

        return self.get(ds_id)

    def get(self, ds_id: str) -> VectorStore:
        """Get existing dataset."""
        return _wrap_mongo_db(self.db, ds_id)

    def delete(self, ds_id: str):
        """Delete dataset."""
        self.db.get_database("vta-rag-vec-db").drop_collection(ds_id)
