import asyncio
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache, wraps
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from pinecone import Pinecone
from neo4j import GraphDatabase
import config
from dataclasses import dataclass
from enum import Enum
import sys
import io
import os
from logging.handlers import RotatingFileHandler

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Setup logging
LOG_DIR = getattr(config, 'LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'hybrid_chat.log')
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 5

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=MAX_LOG_SIZE,
    backupCount=BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# In-memory cache
cache_store = defaultdict(dict)
CACHE_TTL = 3600

def get_cache(key):
    if key in cache_store:
        data, timestamp = cache_store[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        del cache_store[key]
    return None

def set_cache(key, value):
    cache_store[key] = (value, time.time())

@dataclass
class QueryContext:
    raw_query: str
    intent: str
    location: str
    duration: Optional[int] = None
    budget: Optional[str] = None
    preferences: List[str] = None

    def __post_init__(self):
        self.preferences = self.preferences or []

class RetrievalMode(Enum):
    HYBRID = "hybrid"
    SEMANTIC_ONLY = "semantic"
    GRAPH_ONLY = "graph"

# Configuration
EMBED_MODEL = getattr(config, 'EMBED_MODEL', 'text-embedding-3-small')
CHAT_MODEL = getattr(config, 'GPT_MODEL', 'gpt-4o-mini')
TOP_K = 8
INDEX_NAME = getattr(config, 'PINECONE_INDEX', 'vietnam-travel')
NAMESPACE = getattr(config, 'PINECONE_NAMESPACE', 'vietnam')
VECTOR_DIM = getattr(config, 'PINECONE_VECTOR_DIM', 1536)

# Initialize clients
try:
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"OpenAI initialization failed: {e}")
    openai_client = None

try:
    pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
    logger.info("Pinecone client initialized")
except Exception as e:
    logger.error(f"Pinecone initialization failed: {e}")
    pinecone_client = None

try:
    neo4j_driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        max_connection_lifetime=30 * 60
    )
    logger.info("Neo4j driver initialized")
except Exception as e:
    logger.error(f"Neo4j initialization failed: {e}")
    neo4j_driver = None

pinecone_index = None

def cache_key(func_name: str, *args, **kwargs) -> str:
    return f"{func_name}:{hashlib.md5(str(args).encode()).hexdigest()}"

def cached(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key_str = cache_key(func.__name__, *args, **kwargs)
            cached_result = get_cache(cache_key_str)
            if cached_result is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_result
            result = await func(*args, **kwargs)
            set_cache(cache_key_str, result)
            logger.debug(f"Cache set: {func.__name__}")
            return result
        return wrapper
    return decorator

def ensure_pinecone_index():
    global pinecone_index
    if not pinecone_client:
        raise Exception("Pinecone client not initialized")
    try:
        indexes = pinecone_client.list_indexes()
        index_names = [idx.name for idx in indexes]
        if INDEX_NAME not in index_names:
            logger.info(f"Creating Pinecone index: {INDEX_NAME}")
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            for i in range(60):
                time.sleep(5)
                try:
                    desc = pinecone_client.describe_index(INDEX_NAME)
                    if desc.status.get('ready', False):
                        break
                except:
                    continue
            else:
                logger.warning(f"Index {INDEX_NAME} may not be fully ready")
        pinecone_index = pinecone_client.Index(INDEX_NAME)
        logger.info(f"Pinecone index ready: {INDEX_NAME}")
        return pinecone_index
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        raise

try:
    if pinecone_client:
        pinecone_index = ensure_pinecone_index()
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    pinecone_index = None

async def classify_intent(query: str) -> QueryContext:
    if not openai_client:
        return QueryContext(query, "itinerary", "Vietnam", None, None, ["cultural", "romantic"])
    intent_prompt = f"""
    Analyze this travel query and extract structured information in JSON:
    Query: "{query}"
    {{
        "intent": "itinerary|facts|recommendations|comparison",
        "location": "specific city or 'Vietnam'",
        "duration": number of days or null,
        "budget": "budget|midrange|luxury" or null,
        "preferences": ["romantic", "adventure", "cultural", "food"]
    }}
    """
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=200,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            intent_data = json.loads(content[start:end])
        else:
            intent_data = {"intent": "itinerary", "location": "Vietnam"}
        return QueryContext(
            raw_query=query,
            intent=intent_data.get("intent", "itinerary"),
            location=intent_data.get("location", "Vietnam"),
            duration=intent_data.get("duration"),
            budget=intent_data.get("budget"),
            preferences=intent_data.get("preferences", ["cultural", "romantic"])
        )
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}", exc_info=True)
        return QueryContext(query, "itinerary", "Vietnam", None, None, ["cultural", "romantic"])

@lru_cache(maxsize=128)
def get_embedding(text: str) -> List[float]:
    if not openai_client:
        return [0.0] * VECTOR_DIM
    try:
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=[text[:8192]]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        return [0.0] * VECTOR_DIM

@cached(ttl=1800)
async def semantic_search(query: str, top_k: int = TOP_K) -> List[Dict]:
    if not pinecone_index:
        logger.warning("No Pinecone index available")
        return []
    try:
        query_embedding = get_embedding(query)
        if not query_embedding or all(v == 0 for v in query_embedding):
            return []
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k * 2,
            include_metadata=True,
            include_values=False,
            namespace=NAMESPACE
        )
        matches = results.get('matches', [])
        logger.info(f"Pinecone retrieved {len(matches)} matches")
        for match in matches:
            meta = match.get('metadata', {})
            match['enriched'] = {
                'name': meta.get('name', 'Unknown'),
                'type': meta.get('type', 'Entity'),
                'city': meta.get('city', 'Vietnam'),
                'description': meta.get('description', '')[:500]
            }
        return matches
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        return []

@cached(ttl=900)
async def graph_enrichment(node_ids: List[str], context: QueryContext, depth: int = 2) -> List[Dict]:
    if not neo4j_driver or not node_ids:
        logger.warning("No Neo4j or node_ids for graph enrichment")
        return []
    facts = []
    try:
        with neo4j_driver.session() as session:
            depth_limit = min(depth, 3)
            pref_conditions = " AND ".join([f"toLower(target.description) CONTAINS '{pref.lower()}'" 
                                           for pref in context.preferences]) if context.preferences else "TRUE"
            query = f"""
            UNWIND $node_ids as start_id
            MATCH path = (start:Entity {{id: start_id}})-[rels*1..{depth_limit}]-(target:Entity)
            WHERE start.id <> target.id
              AND any(label IN labels(target) WHERE label IN ['Place', 'City', 'Attraction'])
              AND ({pref_conditions})
            WITH start, target, rels,
                 reduce(acc = '', r IN rels | acc + ' -> ' + type(r)) as rel_path,
                 size(rels) as hop_count
            RETURN start.id as source_id,
                   target.id as target_id,
                   COALESCE(target.name, '') as target_name,
                   SUBSTRING(COALESCE(target.description, ''), 0, 300) as target_description,
                   labels(target) as target_labels,
                   type(rels[0]) as primary_rel_type,
                   rel_path as full_rel_path,
                   hop_count,
                   COALESCE(target.type, 'Entity') as target_type
            ORDER BY hop_count ASC
            LIMIT 15
            """
            result = session.run(query, node_ids=node_ids)
            for record in result:
                facts.append({
                    'source_id': record['source_id'],
                    'target_id': record['target_id'],
                    'target_name': record['target_name'],
                    'target_description': record['target_description'],
                    'target_labels': record['target_labels'],
                    'primary_rel_type': record['primary_rel_type'],
                    'full_rel_path': record['full_rel_path'],
                    'hop_count': record['hop_count'],
                    'target_type': record['target_type']
                })
        logger.info(f"Graph retrieved {len(facts)} relationships")
        return facts
    except Exception as e:
        logger.error(f"Graph enrichment failed: {e}", exc_info=True)
        return [{'source_id': nid, 'target_id': nid, 'target_name': f'Location_{nid}', 
                'primary_rel_type': 'SELF'} for nid in node_ids[:5]]

@cached(ttl=900)
async def advanced_reranking(query: str, pinecone_matches: List[Dict], 
                           graph_facts: List[Dict], context: QueryContext) -> Tuple[List[Dict], List[Dict]]:
    if not pinecone_matches:
        return [], graph_facts
    graph_nodes = {f['target_id'] for f in graph_facts}
    scored_matches = []
    for match in pinecone_matches:
        base_score = match.get('score', 0)
        graph_boost = 0.4 if match['id'] in graph_nodes else 0
        meta = match.get('metadata', {})
        type_boost = 0.3 if meta.get('type') in ['Place', 'Attraction'] else 0.1
        pref_boost = sum(0.15 for pref in context.preferences 
                        if pref.lower() in meta.get('description', '').lower()) if context.preferences else 0
        composite_score = (base_score * 0.5 + graph_boost * 0.3 + type_boost * 0.1 + pref_boost * 0.1)
        match['rerank_score'] = composite_score
        scored_matches.append(match)
    reranked = sorted(scored_matches, key=lambda x: x['rerank_score'], reverse=True)[:TOP_K]
    relevant_graph = [f for f in graph_facts 
                     if any(m['id'] == f['target_id'] for m in reranked)]
    return reranked, relevant_graph

async def keyword_search_fallback(context: QueryContext) -> List[str]:
    if not neo4j_driver:
        return []
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS $keyword 
               OR toLower(n.description) CONTAINS $keyword
            RETURN n.id
            LIMIT 10
            """
            result = session.run(query, keyword=context.location.lower())
            return [record['n.id'] for record in result]
    except Exception as e:
        logger.error(f"Fallback search failed: {e}", exc_info=True)
        return []

async def search_summary(reranked_matches: List[Dict], graph_facts: List[Dict]) -> str:
    summary = ["Top Locations:"]
    for match in reranked_matches[:3]:
        meta = match.get('metadata', {})
        summary.append(f"- {meta.get('name', 'Unknown')} ({meta.get('type', 'Entity')}, "
                      f"Score: {match.get('rerank_score', 0):.3f})")
    if graph_facts:
        summary.append("Key Connections:")
        for fact in graph_facts[:2]:
            summary.append(f"- {fact['target_name']} ({fact['primary_rel_type']})")
    return "\n".join(summary)

# Agentic Workflow
class TravelAgent:
    def __init__(self):
        self.openai_client = openai_client
    
    async def plan_itinerary(self, context: QueryContext, retrieval_results: Dict, user_name: str = "Traveler") -> str:
        if not self.openai_client:
            return f"Hi {user_name}, I'm sorry, but travel planning is temporarily unavailable."
        system_prompt = f"""
        You are Odyssey, Vietnam's most delightful and expert travel AI guide, dedicated to crafting unforgettable journeys.
        Your tone is warm, friendly, and professional, like a trusted travel companion who knows Vietnam inside out.
        CAPABILITIES:
        - Create multi-day itineraries tailored to the user's preferences and constraints
        - Ensure geographic feasibility using graph relationships
        - Offer personalized, engaging recommendations
        - Provide practical travel tips (transport, costs, safety, local customs)
        - Make the user feel special by addressing them by name and infusing enthusiasm
        PROCESS:
        1. Start with a warm greeting: "Hi [user_name], let's plan your dream Vietnam adventure!"
        2. Analyze user intent, preferences, and constraints
        3. Prioritize locations based on relevance and scores
        4. Build logical, enjoyable travel paths
        5. Balance activities for a mix of relaxation, exploration, and excitement
        6. Include transport options, budget tips, and safety advice
        7. Cite sources [NodeID] for transparency
        OUTPUT FORMAT:
        Hi [user_name], let's plan your dream Vietnam adventure!
        ## Day X: [Theme]
        ### Morning: [Activity] [NodeID]
        - [Details, why it's special, practical tips]
        ### Afternoon: [Activity] [NodeID]
        - [Details, why it's special, practical tips]
        ### Evening: [Activity] [NodeID]
        - [Details, why it's special, practical tips]
        **Pro Tips:** [Transport, budget, safety, or cultural advice]
        **Why This Works:** [How this day aligns with preferences and intent]
        Wrap up with an encouraging note to make [user_name] excited for their trip!
        """
        context_str = self._build_context(context, retrieval_results)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Query: {context.raw_query}
            Duration: {context.duration} days
            Preferences: {', '.join(context.preferences)}
            Retrieval Context:
            {context_str}
            User Name: {user_name}
            CREATE A DETAILED, FEASIBLE ITINERARY with citations [NodeID]
            """}
        ]
        try:
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=2000,
                temperature=0.2,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Agent planning failed for {user_name}: {e}", exc_info=True)
            return f"Hi {user_name}, I'm sorry, but I encountered an error creating your itinerary. Please try again!"
    
    def _build_context(self, context: QueryContext, results: Dict) -> str:
        ctx_parts = []
        if results.get('semantic_matches'):
            ctx_parts.append("TOP LOCATIONS:")
            for match in results['semantic_matches'][:4]:
                meta = match.get('metadata', {})
                ctx_parts.append(f"- [Node{match['id']}] {meta.get('name', '')} "
                               f"({meta.get('type', '')}) - Score: {match.get('rerank_score', 0):.3f}")
        if results.get('graph_facts'):
            ctx_parts.append("KEY CONNECTIONS:")
            for fact in results['graph_facts'][:6]:
                ctx_parts.append(f"- [Node{fact['source_id']}] --{fact['primary_rel_type']}--> "
                               f"[Node{fact['target_id']}] {fact['target_name']}")
        return "\n".join(ctx_parts)

async def process_travel_query(query: str, user_name: str = "Traveler") -> Dict:
    logger.info(f"Processing query for {user_name}: {query}")
    context = await classify_intent(query)
    semantic_results = []
    graph_results = []
    if pinecone_index:
        semantic_task = asyncio.create_task(semantic_search(context.raw_query))
        fallback_ids_task = asyncio.create_task(keyword_search_fallback(context))
        semantic_results, fallback_ids = await asyncio.gather(semantic_task, fallback_ids_task)
        graph_results = await graph_enrichment(fallback_ids + [m['id'] for m in semantic_results[:3]], context)
    else:
        fallback_ids = await keyword_search_fallback(context)
        graph_results = await graph_enrichment(fallback_ids, context)
    reranked_matches, relevant_graph = await advanced_reranking(
        context.raw_query, semantic_results, graph_results, context
    )
    summary = await search_summary(reranked_matches, relevant_graph)
    agent = TravelAgent()
    itinerary = await agent.plan_itinerary(
        context, {
            'semantic_matches': reranked_matches,
            'graph_facts': relevant_graph,
            'query_context': context.__dict__
        }, user_name
    )
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'semantic_count': len(semantic_results),
        'graph_count': len(graph_results),
        'final_matches': len(reranked_matches),
        'intent': context.intent
    }
    return {
        'itinerary': itinerary,
        'context': context,
        'metrics': metrics,
        'retrieval': {
            'semantic_matches': reranked_matches[:3],
            'graph_facts': relevant_graph[:5]
        },
        'summary': summary
    }

async def interactive_chat():
    print("""
╔══════════════════════════════════════════════════════╗
║ BLUE ENIGMA HYBRID TRAVEL ASSISTANT v2.1             ║
║ Your Personal Vietnam Travel Companion                ║
║ Type 'exit', 'quit', or Ctrl+C to end                ║
╚══════════════════════════════════════════════════════╝
    """)
    user_name = input("🌟 Hello, traveler! What's your name? ").strip()
    if not user_name:
        user_name = "Traveler"
    print(f"\nHi {user_name}, I'm Odyssey, your friendly guide to Vietnam! 💡 Try: 'Create a romantic 4-day itinerary for Vietnam' or 'Best food in Hanoi'\n")
    while True:
        try:
            query = input(f"\n🌍 {user_name}, what's your travel question? ").strip()
            if not query or query.lower() in ('exit', 'quit', 'bye'):
                print(f"\n👋 Safe travels, {user_name}! Thanks for exploring with Odyssey.")
                break
            if len(query) < 3:
                print(f"❌ {user_name}, please enter a valid travel question.")
                continue
            print(f"\n🔄 Hi {user_name}, processing your request... (10-30 seconds)")
            print("📡 Running semantic search + graph analysis + AI planning...")
            start_time = time.time()
            result = await process_travel_query(query, user_name)
            duration = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"🎯 YOUR PERSONALIZED TRAVEL PLAN, {user_name.upper()}!")
            print(f"{'='*70}")
            print(f"⏱️ Response time: {duration:.1f}s")
            print(f"📊 Retrieved: {result['metrics']['final_matches']} key locations")
            print(f"🎯 Intent: {result['context'].intent}")
            print("\n🔍 Search Summary:")
            print(result['summary'])
            print("\n📋 Your Itinerary:")
            print(result['itinerary'])
            print(f"\n{'='*70}")
        except KeyboardInterrupt:
            print(f"\n\n👋 Session interrupted, {user_name}. Safe travels!")
            break
        except Exception as e:
            logger.error(f"Chat processing error for {user_name}: {e}", exc_info=True)
            print(f"❌ Hi {user_name}, an error occurred: {str(e)}")
            print("Please try again.")

def close_connections():
    try:
        if neo4j_driver:
            neo4j_driver.close()
            logger.info("Neo4j connection closed")
    except Exception as e:
        logger.warning(f"Error closing Neo4j: {e}", exc_info=True)
    cache_store.clear()
    logger.info("In-memory cache cleared")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        close_connections()