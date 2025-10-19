"""
🚀 Neo4j Graph Visualization - Neo4j 5.x Compatible with Pyvis
Generates interactive graph visualization for Neo4j subgraph with customizable filters
"""

import asyncio
from neo4j import AsyncGraphDatabase
from pyvis.network import Network
import config
import os
import webbrowser
import pyvis
import logging
from typing import List, Dict, Any

# Setup logging
LOG_DIR = 'logs'  # Define logs directory
os.makedirs(LOG_DIR, exist_ok=True)  # Create logs directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'neo4j_viz.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
NEO_BATCH = getattr(config, 'NEO_BATCH', 200)
OUTPUT_HTML = getattr(config, 'VIZ_OUTPUT_HTML', 'neo4j_viz.html')
LABEL_COLORS = getattr(config, 'LABEL_COLORS', {
    "Place": "#97C2FC",  # Blue
    "City": "#97C2FC",   # Blue
    "Attraction": "#97C2FC",  # Blue
    "Person": "#FF9B9B",  # Red
    "Organization": "#FFD700",  # Gold
    "Event": "#98FB98",  # Pale green
    "Other": "#C0C0C0"  # Silver
})
NETWORK_OPTIONS = getattr(config, 'NETWORK_OPTIONS', '''
{
    "physics": {
        "enabled": true,
        "barnesHut": {
            "gravitationalConstant": -1500,
            "centralGravity": 0.5,
            "springLength": 60,
            "springConstant": 0.06,
            "damping": 0.07
        },
        "stabilization": {
            "iterations": 250
        }
    },
    "layout": {
        "hierarchical": false
    },
    "edges": {
        "smooth": {
            "type": "curvedCW",
            "roundness": 0.1
        }
    }
}
''')

async def fetch_subgraph(tx, limit: int = NEO_BATCH, start_id: str = None, 
                        labels: List[str] = None, preferences: List[str] = None) -> List[Dict[str, Any]]:
    """Fetch a subgraph with optional label and preference filters."""
    labels = labels or []  # Allow all labels if none specified
    preferences = preferences or []
    label_filter = (f"ANY(label IN labels(a) WHERE label IN {labels}) AND ANY(label IN labels(b) WHERE label IN {labels})"
                   if labels else "TRUE")
    pref_filter = " AND ".join([f"toLower(COALESCE(a.description, '')) CONTAINS '{p.lower()}' OR "
                              f"toLower(COALESCE(b.description, '')) CONTAINS '{p.lower()}'"
                              for p in preferences]) if preferences else "TRUE"
    query = f"""
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE ($start_id IS NULL OR a.id = $start_id)
      AND {label_filter}
      AND {pref_filter}
    RETURN a.id AS a_id, 
           labels(a) AS a_labels, 
           a.name AS a_name,
           properties(a) AS a_props,
           b.id AS b_id, 
           labels(b) AS b_labels, 
           b.name AS b_name,
           properties(b) AS b_props,
           type(r) AS rel_type
    LIMIT $limit
    """
    try:
        logger.debug(f"Executing query: {query} with limit={limit}, start_id={start_id}, labels={labels}, preferences={preferences}")
        result = await tx.run(query, limit=limit, start_id=start_id)
        records = []
        async for record in result:  # Compatible with Neo4j driver < 5.8
            records.append(dict(record))
        if not records:
            logger.warning("No relationships found for query")
            logger.debug(f"Query parameters: limit={limit}, start_id={start_id}, labels={labels}, preferences={preferences}")
        else:
            logger.info(f"Fetched {len(records)} relationships from Neo4j")
        return records
    except Exception as e:
        logger.error(f"Error fetching subgraph: {e}", exc_info=True)
        return []

def add_nodes(net: Network, rows: List[Dict[str, Any]], nodes_added: set) -> None:
    """Add nodes to the Pyvis network."""
    for rec in rows:
        for node, id_key, labels_key, name_key, props_key in [
            (rec, "a_id", "a_labels", "a_name", "a_props"),
            (rec, "b_id", "b_labels", "b_name", "b_props")
        ]:
            node_id = rec[id_key]
            if node_id not in nodes_added:
                name = rec[name_key] or node_id
                labels = rec[labels_key]
                primary_label = next((label for label in labels if label != "Entity"), "Other")
                title = f"ID: {node_id}\nName: {name}\nLabels: {', '.join(labels)}"
                props = rec[props_key]
                if props and len(props) > 1:
                    extra_props = "\n".join([f"{k}: {v}" for k, v in list(props.items())[:5]])
                    title += f"\nProperties:\n{extra_props}"
                net.add_node(
                    node_id,
                    label=name,
                    title=title,
                    color=LABEL_COLORS.get(primary_label, LABEL_COLORS["Other"]),
                    size=25,
                    font={"size": 12, "face": "arial"}
                )
                nodes_added.add(node_id)

def add_edges(net: Network, rows: List[Dict[str, Any]], edges_added: set) -> None:
    """Add edges to the Pyvis network."""
    for rec in rows:
        edge_key = (rec["a_id"], rec["b_id"], rec["rel_type"])
        if edge_key not in edges_added:
            net.add_edge(
                rec["a_id"],
                rec["b_id"],
                label=rec["rel_type"],
                title=rec["rel_type"],
                color="#555555",
                width=2,
                smooth={"type": "curvedCW", "roundness": 0.1}
            )
            edges_added.add(edge_key)

async def build_pyvis_network(rows: List[Dict[str, Any]], output_html: str = OUTPUT_HTML) -> None:
    """Build and save interactive graph visualization."""
    logger.info(f"Building visualization with {len(rows)} relationships...")
    if pyvis.__version__ < "0.3.2":
        logger.warning("pyvis version < 0.3.2; consider upgrading: pip install --upgrade pyvis")
    
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f5f5f5",
        font_color="#333333"
    )
    nodes_added = set()
    edges_added = set()
    add_nodes(net, rows, nodes_added)
    add_edges(net, rows, edges_added)
    net.set_options(NETWORK_OPTIONS)
    logger.info(f"Nodes added: {len(net.nodes)}, Edges added: {len(net.edges)}")
    
    try:
        net.save_graph(output_html)
        legend_html = """
        <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.2); z-index: 1000;">
            <h3 style="margin: 0 0 10px 0;">Legend</h3>
            <div><span style="color: #97C2FC;">&#9679;</span> Place/City/Attraction</div>
            <div><span style="color: #FF9B9B;">&#9679;</span> Person</div>
            <div><span style="color: #FFD700;">&#9679;</span> Organization</div>
            <div><span style="color: #98FB98;">&#9679;</span> Event</div>
            <div><span style="color: #C0C0C0;">&#9679;</span> Other</div>
        </div>
        """
        with open(output_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        html_content = html_content.replace('<body>', f'<body>{legend_html}')
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Visualization saved to {os.path.abspath(output_html)} with legend")
        webbrowser.open(f'file://{os.path.abspath(output_html)}')
    except Exception as e:
        logger.error(f"Could not save visualization: {e}")
        try:
            net.save_graph(output_html)
            logger.info(f"Fallback save successful: {os.path.abspath(output_html)}")
            webbrowser.open(f'file://{os.path.abspath(output_html)}')
        except Exception as fallback_e:
            logger.error(f"Fallback save failed: {fallback_e}")

async def verify_connection(tx) -> bool:
    """Verify Neo4j connection and data existence."""
    try:
        result = await tx.run("MATCH (n:Entity) RETURN count(n) as count LIMIT 1")
        count = (await result.single())["count"]
        logger.info(f"Found {count} Entity nodes in database")
        return count > 0
    except Exception as e:
        logger.error(f"Error verifying connection: {e}")
        return False

async def debug_schema(tx) -> Dict[str, Any]:
    """Debug database schema and data."""
    try:
        schema_info = {}
        # Node counts by label
        label_query = "MATCH (n) RETURN labels(n) AS labels, count(n) AS count"
        label_result = await tx.run(label_query)
        schema_info["labels"] = {tuple(record["labels"]): record["count"] async for record in label_result}
        
        # Relationship types
        rel_query = "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
        rel_result = await tx.run(rel_query)
        schema_info["relationships"] = {record["type"]: record["count"] async for record in rel_result}
        
        # Sample nodes with description
        sample_query = "MATCH (n:Entity) WHERE n.description IS NOT NULL RETURN n.description LIMIT 5"
        sample_result = await tx.run(sample_query)
        schema_info["sample_descriptions"] = [record["n.description"] async for record in sample_result]
        
        logger.info(f"Schema debug: {schema_info}")
        return schema_info
    except Exception as e:
        logger.error(f"Schema debug failed: {e}")
        return {}

async def main():
    """Main async function for visualization."""
    driver = None
    try:
        driver = AsyncGraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        async with driver.session() as session:
            if not await session.execute_read(verify_connection):
                logger.error("No Entity nodes found. Run neo4j_loader.py first.")
                return
            logger.info("Fetching subgraph...")
            # First attempt with relaxed filters
            rows = await session.execute_read(
                fetch_subgraph,
                limit=NEO_BATCH,
                labels=None,  # Allow all labels
                preferences=None  # No preference filters
            )
            if not rows:
                logger.error("No relationships found. Debugging schema...")
                schema_info = await session.execute_read(debug_schema)
                logger.info(f"Schema info: {schema_info}")
                return
            await build_pyvis_network(rows, OUTPUT_HTML)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if driver:
            await driver.close()
            logger.info("Neo4j driver closed cleanly.")

if __name__ == "__main__":
    asyncio.run(main())