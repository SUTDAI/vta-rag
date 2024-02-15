"""Global constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["TESTING_DIR"]

load_dotenv()

TESTING_DIR = Path("./testing/").absolute()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_EP = os.getenv("MONGO_EP")
MONGO_GROUP_ID = os.getenv("MONGO_GROUP_ID")
MONGO_CLUSTER_NAME = os.getenv("MONGO_CLUSTER_NAME")

MONGO_ADMIN_USER = os.getenv("MONGO_ADMIN_USER")
MONGO_ADMIN_KEY = os.getenv("MONGO_ADMIN_KEY")
MONGO_VEC_DB_NAME = os.getenv("MONGO_VEC_DB_NAME")
