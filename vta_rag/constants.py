"""Global constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOCAL_TEST = os.getenv("LOCAL_TEST", "0") == "1"
TESTING_DIR = Path(os.getenv("TESTING_DIR", ".")).absolute()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority",
)
MONGO_EP = os.getenv("MONGO_EP", "https://cloud.mongodb.com/api/atlas/v2")
MONGO_GROUP_ID = os.getenv("MONGO_GROUP_ID", "000000000000000000000000")
MONGO_CLUSTER_NAME = os.getenv("MONGO_CLUSTER_NAME", "cluster")

MONGO_ADMIN_USER = os.getenv("MONGO_ADMIN_USER", "aaaaaaaa")
MONGO_ADMIN_KEY = os.getenv("MONGO_ADMIN_KEY", "00000000-0000-0000-0000-000000000000")
MONGO_VEC_DB_NAME = os.getenv("MONGO_VEC_DB_NAME", "database")
