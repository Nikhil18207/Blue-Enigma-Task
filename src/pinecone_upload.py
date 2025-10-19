import json
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
import config
import logging
from typing import List, Dict, Any, Tuple
import time
import os
from logging.handlers import RotatingFileHandler

# Setup logging
LOG_DIR = getattr(config, 'LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'pinecone_upload.log')
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 5  # Keep 5 backup log files

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=MAX_LOG_SIZE,
    backupCount=BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Clear any existing handlers to avoid duplicates
logger.handlers.clear()

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Config from config.py
DATA_FILE = getattr(config, 'DATA_FILE', 'data/vietnam_travel_dataset.json')
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 64)
MAX_CONCURRENT_BATCHES = getattr(config, 'MAX_CONCURRENT_BATCHES', 4)
INDEX_NAME = config.PINECONE_INDEX
VECTOR_DIM = config.PINECONE_VECTOR_DIM
NAMESPACE = getattr(config, 'PINECONE_NAMESPACE', 'vietnam')
EMBED_MODEL = getattr(config, 'EMBED_MODEL', 'text-embedding-3-small')

# Initialize clients
try:
    openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"OpenAI initialization failed: {e}", exc_info=True)
    openai_client = None

try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    logger.info("Pinecone client initialized")
except Exception as e:
    logger.error(f"Pinecone initialization failed: {e}", exc_info=True)
    pc = None

# Pinecone Index Setup
def ensure_index_ready():
    if not pc:
        raise Exception("Pinecone client not initialized")
    try:
        indexes = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in indexes:
            logger.info(f"[INIT] Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        for attempt in range(12):
            desc = pc.describe_index(INDEX_NAME)
            if desc.status.get('ready', False):
                logger.info(f"[SUCCESS] Index {INDEX_NAME} is ready")
                return pc.Index(INDEX_NAME)
            logger.info(f"[WAIT] Index not ready (attempt {attempt + 1}/12)")
            time.sleep(5)
        raise Exception(f"Index {INDEX_NAME} failed to become ready")
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}", exc_info=True)
        raise

try:
    index = ensure_index_ready()
except Exception as e:
    logger.error(f"Failed to initialize Pinecone index: {e}", exc_info=True)
    index = None

# Enhanced Embedding Functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
async def get_embeddings_batch(texts: List[str], session: aiohttp.ClientSession) -> List[List[float]]:
    if not texts:
        return []
    try:
        resp = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[:2048]
        )
        embeddings = [data.embedding for data in resp.data]
        if embeddings and len(embeddings[0]) != VECTOR_DIM:
            raise ValueError(f"Expected {VECTOR_DIM} dims, got {len(embeddings[0])}")
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding batch failed: {e}", exc_info=True)
        raise

async def process_batch(batch_items: List[Tuple[str, str, Dict]], session: aiohttp.ClientSession, 
                      semaphore: asyncio.Semaphore) -> int:
    async with semaphore:
        try:
            texts = [item[1] for item in batch_items]
            embeddings = await get_embeddings_batch(texts, session)
            if not embeddings:
                return 0
            vectors = []
            for (item_id, _, meta), embedding in zip(batch_items, embeddings):
                vectors.append({
                    "id": str(item_id),
                    "values": embedding,
                    "metadata": {
                        **meta,
                        "text_length": len(meta.get('description', '')),
                        "embedding_model": EMBED_MODEL,
                        "timestamp": time.time(),
                        "namespace": NAMESPACE
                    }
                })
            if vectors:
                upsert_response = index.upsert(vectors=vectors, namespace=NAMESPACE)
                logger.debug(f"Upserted {len(vectors)} vectors")
                await asyncio.sleep(0.1)
                return len(vectors)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 0

async def upload_in_parallel(batches: List[List[Tuple[str, str, Dict]]], session: aiohttp.ClientSession, 
                           semaphore: asyncio.Semaphore):
    tasks = [process_batch(batch, session, semaphore) for batch in batches]
    results = await tqdm_asyncio.gather(*tasks, desc="Uploading batches")
    total_uploaded = sum(r for r in results if r > 0)
    logger.info(f"Completed parallel upload: {total_uploaded} vectors")
    return total_uploaded

# Data Processing
def load_and_prepare_data() -> List[Tuple[str, str, Dict]]:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        items = []
        skipped = 0
        for node in nodes:
            semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1500]
            if not semantic_text.strip():
                skipped += 1
                continue
            meta = {
                "id": node.get("id"),
                "type": node.get("type", "unknown"),
                "name": node.get("name", "Unknown"),
                "city": node.get("city", node.get("region", "Vietnam")),
                "tags": node.get("tags", []),
                "description": semantic_text[:500]
            }
            items.append((node["id"], semantic_text, meta))
        logger.info(f"Loaded {len(items)} valid items (skipped {skipped})")
        return items
    except FileNotFoundError:
        logger.error(f"Dataset not found: {DATA_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {DATA_FILE}: {e}")
        raise

def chunk_data(items: List[Tuple[str, str, Dict]], batch_size: int) -> List[List[Tuple[str, str, Dict]]]:
    batch_size = min(batch_size, len(items))
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# Main Pipeline
async def main():
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        items = load_and_prepare_data()
        if not items:
            logger.error("No valid data to upload")
            return
        batches = chunk_data(items, BATCH_SIZE)
        logger.info(f"Created {len(batches)} batches of size up to {BATCH_SIZE}")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
        total_uploaded = await upload_in_parallel(batches, session, semaphore)
        elapsed = time.time() - start_time
        upserts_per_second = total_uploaded / elapsed if elapsed > 0 else 0
        logger.info(f"Upload complete")
        logger.info(f"Total vectors uploaded: {total_uploaded}")
        logger.info(f"Total time: {elapsed:.2f}s ({upserts_per_second:.1f} vectors/sec)")
        logger.info(f"Namespace: '{NAMESPACE}' in index '{INDEX_NAME}'")
        logger.info(f"Take screenshots of:")
        logger.info(f"   1. This terminal output")
        logger.info(f"   2. Pinecone dashboard showing record count in '{NAMESPACE}' namespace")
        logger.info(f"   3. Verify with: python hybrid_chat.py")

# Cleanup
def cleanup():
    logger.info("Cleaning up resources")
    cache_store = globals().get('cache_store', None)
    if cache_store:
        cache_store.clear()
        logger.info("Cache cleared")

# Entry Point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Upload interrupted by user")
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise
    finally:
        cleanup() 