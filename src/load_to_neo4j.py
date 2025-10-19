import json
import logging
import os
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase
from tqdm.asyncio import tqdm_asyncio
import config
from datetime import datetime
import asyncio
from logging.handlers import RotatingFileHandler

# Enhanced logging
LOG_DIR = getattr(config, 'LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'neo4j_load.log')
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

# Config
DATA_FILE = getattr(config, 'DATA_FILE', os.path.join("data", "vietnam_travel_dataset.json"))
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 100)
MAX_CONNECTION_POOL = 50

def sanitize_label(label: str) -> str:
    """Sanitize labels for Cypher"""
    if not label or label == "None":
        return "Entity"
    safe = "".join(c for c in str(label) if c.isalnum() or c in ['_']).title()
    return safe[:50] or "Entity"

def sanitize_rel_type(rel_type: str) -> str:
    """Sanitize and map relationship types"""
    if not rel_type:
        return "RELATED_TO"
    rel_map = {
        "near": "NEAR",
        "part_of": "PART_OF",
        "contains": "CONTAINS",
        "connected_to": "CONNECTED_TO",
        "located_in": "LOCATED_IN",
        "available_in": "AVAILABLE_IN"
    }
    safe = "".join(c for c in str(rel_type) if c.isalnum() or c in ['_']).upper()
    return rel_map.get(rel_type.lower(), safe or "RELATED_TO")

async def setup_constraints_safely(driver):
    """Async constraint and index setup for Neo4j 5.x"""
    neo4j_5_operations = [
        "CREATE CONSTRAINT entity_id_constraint IF NOT EXISTS FOR (n:Entity) REQUIRE (n.id) IS UNIQUE",
        "CREATE CONSTRAINT place_id_constraint IF NOT EXISTS FOR (n:Place) REQUIRE (n.id) IS UNIQUE",
        "CREATE CONSTRAINT city_id_constraint IF NOT EXISTS FOR (n:City) REQUIRE (n.id) IS UNIQUE",
        "CREATE CONSTRAINT attraction_id_constraint IF NOT EXISTS FOR (n:Attraction) REQUIRE (n.id) IS UNIQUE",
        "CREATE INDEX entity_city_index IF NOT EXISTS FOR (n:Entity) ON (n.city)",
        "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        "CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.type)"
    ]
    success_count = 0
    async with driver.session() as session:
        for op in neo4j_5_operations:
            try:
                await session.run(op)
                constraint_name = op.split()[3] if 'CONSTRAINT' in op else op.split()[2]
                logger.info(f"Created: {constraint_name}")
                success_count += 1
            except Exception as e:
                if "already exists" in str(e).lower():
                    constraint_name = op.split()[3] if 'CONSTRAINT' in op else op.split()[2]
                    logger.debug(f"Constraint/index exists: {constraint_name}")
                else:
                    logger.warning(f"Constraint/index skipped: {e}", exc_info=True)
    logger.info(f"Setup: {success_count}/{len(neo4j_5_operations)} operations")
    return True

def prepare_node_properties(node: Dict[str, Any]) -> Dict[str, Any]:
    """Clean properties for Neo4j"""
    exclude_fields = {"connections", "embedding", "raw_data"}
    props = {
        k: str(v)[:2000] if isinstance(v, (dict, list)) else v[:2000] if isinstance(v, str) else v
        for k, v in node.items() if k not in exclude_fields
    }
    props.setdefault("id", node.get("id", "unknown"))
    props.setdefault("type", node.get("type", "Entity"))
    props.setdefault("name", node.get("name", "Unnamed Entity"))
    props["processed_at"] = datetime.now().isoformat()
    return props

async def batch_upsert_nodes(tx, batch_nodes: List[Dict[str, Any]]) -> int:
    """Async batch upsert with dynamic labels"""
    if not batch_nodes:
        return 0
    node_data = []
    for node in batch_nodes:
        if not node.get("id"):
            continue
        props = prepare_node_properties(node)
        labels = ["Entity"]
        node_type = node.get("type", "").lower()
        if node_type in ["place", "attraction", "city"]:
            labels.append(sanitize_label(node_type))
            logger.debug(f"Node type: {node_type} for id: {props['id']}")
        node_data.append({
            "id": props["id"],
            "props": props,
            "labels": labels
        })
    if not node_data:
        return 0
    query = """
    UNWIND $nodes as node_data
    MERGE (n:Entity {id: node_data.id})
    SET n += node_data.props
    FOREACH (label IN node_data.labels[1..] | SET n:`+label+`)
    RETURN count(n) as created
    """
    try:
        result = await tx.run(query, nodes=node_data)
        return (await result.single())["created"]
    except Exception as e:
        logger.warning(f"Bulk upsert failed: {e}", exc_info=True)
        created = 0
        for node_data in node_data:
            try:
                await tx.run("""
                    MERGE (n:Entity {id: $id})
                    SET n += $props
                    FOREACH (label IN $labels[1..] | SET n:`+label+`)
                """, id=node_data["id"], props=node_data["props"], labels=node_data["labels"])
                created += 1
            except:
                pass
        return created

async def batch_create_relationships(tx, batch_rels: List[Dict[str, Any]]) -> int:
    """Async relationship creation with dynamic types using MERGE"""
    if not batch_rels:
        return 0
    valid_rels = []
    seen_rels = set()  # Track unique relationships to prevent duplicates
    for rel_data in batch_rels:
        source_id = rel_data.get("source_id")
        rel = rel_data.get("rel", {})
        rel_type = sanitize_rel_type(rel.get("relation", "RELATED_TO"))
        target_id = rel.get("target")
        if source_id and target_id and source_id != target_id:
            rel_key = f"{source_id}-{rel_type}-{target_id}"
            if rel_key not in seen_rels:
                valid_rels.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    "props": {"created_at": datetime.now().isoformat()}
                })
                seen_rels.add(rel_key)
                logger.debug(f"Preparing relationship: {source_id} -[{rel_type}]-> {target_id}")
    if not valid_rels:
        logger.warning("No valid relationships to create")
        return 0
    query = """
    UNWIND $relationships as rel_data
    MATCH (a:Entity {id: rel_data.source_id})
    MATCH (b:Entity {id: rel_data.target_id})
    WHERE a IS NOT NULL AND b IS NOT NULL
    MERGE (a)-[r:`+rel_data.rel_type+` {created_at: rel_data.props.created_at}]->(b)
    RETURN count(r) as created
    """
    try:
        result = await tx.run(query, relationships=valid_rels)
        count = (await result.single())["created"]
        logger.debug(f"Created {count} relationships in batch")
        return count
    except Exception as e:
        logger.warning(f"Bulk relationships failed: {e}, falling back to individual", exc_info=True)
        created = 0
        for rel in valid_rels:
            try:
                await tx.run("""
                    MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                    MERGE (a)-[r:`+rel.rel_type+` {created_at: $created_at}]->(b)
                    RETURN r
                """, source_id=rel["source_id"], target_id=rel["target_id"], 
                     rel_type=rel["rel_type"], created_at=rel["props"]["created_at"])
                created += 1
                logger.debug(f"Created individual relationship: {rel['source_id']} -[{rel['rel_type']}]-> {rel['target_id']}")
            except Exception as e:
                logger.error(f"Failed to create relationship {rel['source_id']} -[{rel['rel_type']}]-> {rel['target_id']}: {e}", exc_info=True)
        return created

async def validate_data(nodes: List[Dict[str, Any]]) -> tuple[int, int, Dict[str, int]]:
    """Validate data and count relationship types"""
    valid_nodes = sum(1 for n in nodes if n.get("id"))
    relationships = []
    rel_type_counts = {}
    seen_rels = set()  # Track unique relationships
    for node in nodes:
        if node.get("id"):
            for rel in node.get("connections", []):
                if rel.get("target"):
                    rel_key = f"{node['id']}-{rel.get('relation', 'RELATED_TO')}-{rel['target']}"
                    if rel_key not in seen_rels:
                        relationships.append(rel)
                        seen_rels.add(rel_key)
                        rel_type = sanitize_rel_type(rel.get("relation", "RELATED_TO"))
                        rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
    logger.info(f"Validation: {valid_nodes} nodes, {len(relationships)} relationships")
    logger.info(f"Relationship types: {rel_type_counts}")
    return valid_nodes, len(relationships), rel_type_counts

async def verify_load_quality(driver):
    """Verify loaded data with detailed metrics"""
    async with driver.session() as session:
        try:
            node_count = (await (await session.run("MATCH (n:Entity) RETURN count(n)")).single())["count(n)"]
            rel_count = (await (await session.run("MATCH ()-[r]->() RETURN count(r)")).single())["count(r)"]
            place_count = (await (await session.run("MATCH (n:Place) RETURN count(n)")).single())["count(n)"]
            hanoi_count = (await (await session.run("MATCH (n:Entity) WHERE n.city = 'Hanoi' RETURN count(n)")).single())["count(n)"]
            logger.info(f"SUCCESS: {node_count} nodes, {rel_count} relationships")
            logger.info(f"Place nodes: {place_count}")
            logger.info(f"Hanoi coverage: {hanoi_count} entities")
            # Check for duplicate relationships
            dup_query = """
            MATCH (a)-[r]->(b)
            WITH a, b, type(r) AS rel_type, COLLECT(r) AS rels
            WHERE SIZE(rels) > 1
            RETURN COUNT(*) AS duplicate_count
            """
            dup_count = (await (await session.run(dup_query)).single())["duplicate_count"]
            if dup_count > 0:
                logger.warning(f"Found {dup_count} duplicate relationships")
            return {"nodes": node_count, "relationships": rel_count, "places": place_count, "hanoi": hanoi_count, "duplicates": dup_count}
        except Exception as e:
            logger.warning(f"Verification failed: {e}", exc_info=True)
            return {"nodes": 0, "relationships": 0, "places": 0, "hanoi": 0, "duplicates": 0}

async def load_nodes(driver, nodes: List[Dict[str, Any]]) -> int:
    """Async node loading"""
    total_nodes = 0
    async with driver.session() as session:
        for i in tqdm_asyncio(range(0, len(nodes), BATCH_SIZE), desc="Nodes"):
            batch = nodes[i:i + BATCH_SIZE]
            count = await session.execute_write(batch_upsert_nodes, batch)
            total_nodes += count
            logger.info(f"Loaded {count} nodes for batch {i//BATCH_SIZE + 1}")
    return total_nodes

async def load_relationships(driver, all_rels: List[Dict[str, Any]]) -> int:
    """Async relationship loading"""
    total_rels = 0
    async with driver.session() as session:
        for i in tqdm_asyncio(range(0, len(all_rels), BATCH_SIZE), desc="Rels"):
            batch_rels = all_rels[i:i + BATCH_SIZE]
            count = await session.execute_write(batch_create_relationships, batch_rels)
            total_rels += count
            logger.info(f"Loaded {count} relationships for batch {i//BATCH_SIZE + 1}")
    return total_rels

async def main():
    """Main async pipeline"""
    driver = None
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Dataset missing: {DATA_FILE}")
        logger.info(f"Loading {DATA_FILE}")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        logger.info(f"Loaded {len(nodes)} raw nodes")
        valid_count, rel_count, rel_type_counts = await validate_data(nodes)
        driver = AsyncGraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=30 * 60,
            max_connection_pool_size=MAX_CONNECTION_POOL
        )
        logger.info("Attempting constraints (optional)")
        await setup_constraints_safely(driver)
        start_time = datetime.now()
        total_nodes = await load_nodes(driver, nodes)
        all_rels = [{"source_id": node["id"], "rel": rel} for node in nodes if node.get("id") for rel in node.get("connections", [])]
        total_rels = await load_relationships(driver, all_rels)
        elapsed = datetime.now() - start_time
        logger.info(f"LOAD COMPLETE in {elapsed}")
        logger.info(f"{total_nodes} nodes, {total_rels} relationships")
        logger.info(f"Relationship types: {rel_type_counts}")
        quality = await verify_load_quality(driver)
        logger.info("Neo4j READY for hybrid_chat.py")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
    finally:
        if driver:
            await driver.close()
            logger.info("Driver closed")

if __name__ == "__main__":
    asyncio.run(main()) 