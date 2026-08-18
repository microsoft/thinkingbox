#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Import Typesense snapshot files created by export_typesense_snapshot.py
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import httpx
import typesense


def wait_until_healthy(url: str, timeout: float, max_wait: float):
    s = httpx.Client(timeout=timeout)
    start = time.monotonic()
    while True:
        try:
            resp = s.get(url)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            # Typesense may reject connections while it is still starting.
            pass
        if time.monotonic() - start > max_wait:
            raise TimeoutError(
                f"Timed out waiting for {url} to become healthy after {max_wait} seconds"
            )
        time.sleep(1)


def load_json_file(path: str | Path):
    """
    Load a JSON object from a file. If the file ends with '.gz', use gzip to open it.
    Returns the parsed object.
    """
    path = Path(path)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def iter_files_with_suffix(path: str | Path, suffix: str | tuple[str, ...]):
    """
    Yield Path objects for all .jsonl and .jsonl.gz files in a directory and its subdirectories,
    or yield the path itself if it is a .jsonl or .jsonl.gz file.
    """
    path = Path(path)
    if path.is_file() and path.name.endswith(suffix):
        yield path
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and p.name.endswith(suffix):
                yield p
    else:
        raise FileNotFoundError(str(path))


class TypesenseClient:
    def __init__(self, host: str, port: int, api_key: str, timeout: float = 30.0):
        self.client = typesense.Client(
            {
                "api_key": api_key,
                "nodes": [{"host": host, "port": str(port), "protocol": "http"}],
                "connection_timeout_seconds": timeout,
            }
        )

    def ensure_collection(
        self, collection_name: str, schema: dict, replace: bool = False
    ):
        collection_exists = False
        if replace:
            # delete collection if it exists
            try:
                self.client.collections[collection_name].delete()
                print(f"Deleted collection {collection_name}")
            except typesense.exceptions.ObjectNotFound:
                # The requested collection is already absent.
                pass
        else:
            # check if collection already exists
            try:
                self.client.collections[collection_name].retrieve()
                collection_exists = True
                print(f"Found collection {collection_name}")
            except typesense.exceptions.ObjectNotFound:
                # The collection is created below when it cannot be retrieved.
                pass
        # Create collection if it doesn't exist
        if not collection_exists:
            if schema["name"] != collection_name:
                schema = schema.copy()
                schema["name"] = collection_name
            self.client.collections.create(schema)
            print(f"Created collection {collection_name}")

    def import_documents(
        self, collection_name: str, documents: list, batch_size: int = 100
    ):
        # Import documents in batches
        imported = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                # Use upsert instead of create to handle potential duplicates
                self.client.collections[collection_name].documents.import_(
                    batch,
                    {"action": "upsert"},
                )
                imported += len(batch)
            except typesense.exceptions.TypesenseClientError as e:
                raise RuntimeError(
                    f"Error importing batch {i // batch_size + 1}"
                ) from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8108)
    p.add_argument("--inputs", help="file or directory")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--collection", default="thinkingbox_universal_search_tool")
    p.add_argument("--max-wait", type=float, default=300)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--replace", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("TYPESENSE_API_KEY", "Fake")

    wait_until_healthy(
        f"http://{args.host}:{args.port}/health",
        args.timeout,
        args.max_wait,
    )
    client = TypesenseClient(
        host=args.host,
        port=args.port,
        api_key=api_key,
        timeout=args.timeout,
    )

    for i, f in enumerate(
        iter_files_with_suffix(Path(args.inputs), (".json", ".json.gz"))
    ):
        obj = load_json_file(f)
        if i == 0:
            client.ensure_collection(
                args.collection,
                schema=obj["schema"],
                replace=args.replace,
            )
        print(f"Importing {f!s}")
        client.import_documents(
            args.collection,
            documents=obj["documents"],
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
