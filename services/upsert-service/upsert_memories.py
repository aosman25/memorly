import logging
from pathlib import Path
from pymilvus import MilvusClient
from dotenv import load_dotenv
import os
import json
from utils import *

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
MAX_CONTENT_LENGTH = 65535
FAILED_MEMORIES_PATH = "failed_memories.json"

def upsert_memories(
    main_directory,
    collection_name="memories",
    partition_name="_default",
    batch_size=1000,
):
    """
    Upsert memory data into Milvus collection.

    Args:
        main_directory: Directory containing .jsonl files with memory data
        collection_name: Name of the Milvus collection
        partition_name: Name of the partition to upsert into
        batch_size: Number of records to batch before upserting
    """

    def ensure_partition_exists():
        existing_partitions = set(client.list_partitions(collection_name))
        if partition_name not in existing_partitions:
            client.create_partition(
                collection_name=collection_name,
                partition_name=partition_name
            )

    def stream_jsonl(file_path):
        """Stream JSONL file line by line."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Truncate content if too long
                    if "content" in record and isinstance(record["content"], str):
                        record["content"] = record["content"][:MAX_CONTENT_LENGTH]
                    yield record

    def upsert_batch(batch_data):
        """Upsert a batch of data into Milvus."""
        client.upsert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=batch_data,
        )

    def load_failed_memories():
        """Load list of previously failed memory files."""
        try:
            with open(FAILED_MEMORIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_failed_memory(file_name):
        """Save failed memory file to tracking list."""
        failed = load_failed_memories()
        failed.append(str(file_name))
        with open(FAILED_MEMORIES_PATH, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)

    # Create collection if it doesn't exist
    if not client.has_collection(collection_name=collection_name):
        create_memory_collection(client=client, collection_name=collection_name)

    ensure_partition_exists()

    # Find all JSONL files
    jsonl_files = list(Path(main_directory).rglob("*.jsonl"))
    if not jsonl_files:
        logging.warning("No .jsonl files found.")
        return False

    # Process each file
    for file_idx, file_path in enumerate(jsonl_files, start=1):
        logging.info(f"[{file_idx}/{len(jsonl_files)}] Processing: {file_path}")
        batch = []
        count = 0

        try:
            for record in stream_jsonl(file_path):
                batch.append(record)
                if len(batch) >= batch_size:
                    upsert_batch(batch)
                    count += len(batch)
                    logging.info(f"Upserted {count} records so far from {file_path.name}")
                    batch = []

            # Upsert remaining records
            if batch:
                upsert_batch(batch)
                count += len(batch)
                logging.info(f"Final batch upserted from {file_path.name} ({len(batch)} records)")

            logging.info(f"Successfully upserted {count} total records from {file_path.name}")

        except Exception as e:
            logging.error(f"Error while processing {file_path}: {e}")
            save_failed_memory(file_path.name)
            continue  # skip to next file

    logging.info("All data upserted successfully (excluding failed files).")
    return True


if __name__ == "__main__":
    logging.info("Connecting to Milvus Database...")
    client = MilvusClient(
        uri=f"http://{os.getenv('MILVUS_IP')}",
        token=os.getenv("MILVUS_TOKEN")
    )
    logging.info("Successfully Connected to Milvus!")

    # Get data directory from environment or use default
    root_folder = os.getenv("ROOT_FOLDER", "/home/aomsan25/memorly")
    main_folder = os.path.join(root_folder, "data", "memories")

    # Upsert memories
    upsert_memories(
        main_folder,
        collection_name="memories",
        partition_name="_default"
    )
