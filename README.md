# Blue-Enigma-Task

# ODYSSEY - Vietnam Travel AI Guide

![Vietnam Travel AI](https://img.shields.io/badge/ODYSSEY-Vietnam%20AI-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Neo4j](https://img.shields.io/badge/Neo4j-5.x-orange) ![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)

🌟 **Intelligent travel assistant** combining **semantic search**, **knowledge graphs**, and **AI planning** to create **personalized Vietnam travel itineraries**.

---

## 🎯 Key Capabilities

- **Multi-modal Search**: Semantic (vector) + Graph (relationships) + Keyword fallback
- **Personalized Itineraries**: 1-14 day plans based on preferences, budget, duration
- **Interactive Chat**: Real-time conversation with your AI travel companion
- **Rich Visualizations**: Interactive Neo4j graph explorer with Pyvis
- **Production Ready**: Caching, retries, logging, async processing

**🗺️ Coverage**: 500+ Vietnam locations (Hanoi, HCMC, Ha Long Bay, Sapa, etc.)  
**⚡ Performance**: <30s response time with hybrid retrieval

---

## 🚀 Quick Start

```bash
# 1. Clone & Setup
git clone https://github.com/Nikhil18207/Blue-Enigma-Task.git
cd Blue-Enigma-Task

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Configure (copy template)
cp src/config.py src/config_template.py  # Edit with your API keys

# 4. Download Dataset
# Download 'vietnam_travel_dataset.json' and place in 'data/' folder

# 5. Load Data
python src/neo4j_loader.py      # Loads to Neo4j
python src/pinecone_upload.py   # Uploads embeddings to Pinecone

# 6. Run Interactive Chat
python src/hybrid_chat.py       # Start Odyssey AI chat

# 7. Visualize Graph
python src/visualize_graph.py   # Generates neo4j_viz.html
```

**Expected Output**: Interactive terminal chat for travel queries like *"Plan a 5-day romantic trip to Hanoi"*.

---

## 📋 Installation & Setup

### Prerequisites

- **Python 3.9+**
- **Neo4j Database**: [AuraDB (free tier)](https://neo4j.com/cloud/aura/) or local Neo4j 5.x
- **Pinecone Account**: [Sign up for API key](https://www.pinecone.io/)
- **OpenAI API Key**: [Get key](https://platform.openai.com/api-keys)

### Step-by-Step Setup

#### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Configuration
- Copy `src/config.py` and fill in your credentials
- Ensure `data/vietnam_travel_dataset.json` exists

#### 3. Data Loading
```bash
python src/neo4j_loader.py    # → Logs in logs/neo4j_load.log
python src/pinecone_upload.py # → Uploads to Pinecone namespace "vietnam"
```

#### 4. Verification
- Check logs in `logs/`
- Run `python src/visualize_graph.py` to generate `neo4j_viz.html`

---

## 🛠️ Project Structure

```
Blue-Enigma-Task/
├── data/
│   └── vietnam_travel_dataset.json     # Travel dataset (nodes + connections)
├── src/                                # Core source code
│   ├── config.py                       # API keys & settings
│   ├── hybrid_chat.py                  # Main AI chat & itinerary planning
│   ├── neo4j_loader.py                 # Neo4j data loader
│   └── pinecone_upload.py              # Pinecone embedding uploader
├── logs/                               # Runtime logs (auto-created)
├── Screenshots/                        # Proof-of-concept outputs
├── .gitignore                          # Git exclusions
├── requirements.txt                    # Dependencies
├── README.md                           # This file
└── neo4j_viz.html                      # Generated graph visualization
```

---

## 🔍 Core Features

### 1. **Hybrid Retrieval Pipeline** (`hybrid_chat.py`)

- **Intent Classification**: GPT-4o-mini parses queries (itinerary, recommendations, facts)
- **Semantic Search**: Pinecone vector similarity (text-embedding-3-small)
- **Graph Enrichment**: Neo4j traversals (NEAR, CONTAINS relationships)
- **Reranking**: Composite scoring (semantic + graph + preferences)
- **AI Planning**: Detailed itineraries with citations `[NodeID]`

**Example Query**: *"Create a 4-day adventure itinerary in Sapa under $500"*  
**Output**: Day-by-day plan with transport, tips, and budget breakdown.

### 2. **Data Ingestion**

- **Neo4j Loader**: Async batch upserts with constraints
- **Pinecone Uploader**: Parallel embedding generation + upsert with retries

### 3. **Visualization** (`visualize_graph.py`)

- Fetches Neo4j subgraph → Renders interactive Pyvis network
- **Customizable**: Filters by labels/preferences
- **Output**: `neo4j_viz.html` with legend and physics layout

---

## ⚙️ Configuration

Edit `src/config.py`:

```python
# Neo4j
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# OpenAI
OPENAI_API_KEY = "sk-..."
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

# Pinecone
PINECONE_API_KEY = "your-key"
PINECONE_INDEX = "vietnam-travel"
PINECONE_NAMESPACE = "vietnam"
PINECONE_VECTOR_DIM = 1536

# Other
DATA_FILE = "data/vietnam_travel_dataset.json"
BATCH_SIZE = 64
LOG_DIR = "logs"
```

---

## 🧪 Running & Testing

### Interactive Mode
```bash
python src/hybrid_chat.py
```
- Greets user, processes queries in a loop
- Handles intents: itinerary, facts, recommendations
- **Metrics**: Response time, retrieved matches

### Batch Testing
```python
# Add to hybrid_chat.py
result = await process_travel_query("Best beaches in Phu Quoc for families", "TestUser")
print(result['itinerary'])
```

### Debugging
- **Logs**: `logs/hybrid_chat.log`, `logs/pinecone_upload.log`
- **Cache**: In-memory TTL=1h for embeddings/searches
- **Fallbacks**: Keyword search if Pinecone/Neo4j unavailable

---

## 📈 Performance & Metrics

| Component          | Throughput       | Latency | Notes                  |
|--------------------|------------------|---------|------------------------|
| Embedding Batch    | 64 items/sec     | 2-5s    | OpenAI async           |
| Neo4j Upsert       | 100 nodes/batch  | 1-3s    | Async with retries     |
| Pinecone Query     | TOP_K=8          | <1s     | Cosine similarity      |
| Full Query         | End-to-End       | 10-30s  | Includes planning      |

**Dataset Size**: ~500 entities, 1000+ relationships  
**Scalability**: Semaphores limit concurrency to avoid rate limits

---

## 🤝 Contributing

1. Fork the repo & create a feature branch:
   ```bash
   git checkout -b feat/amazing-feature
   ```

2. Commit changes:
   ```bash
   git commit -m 'Add some amazing feature'
   ```

3. Push to branch:
   ```bash
   git push origin feat/amazing-feature
   ```

4. Open Pull Request

### Guidelines
- Follow [PEP8 style](https://www.python.org/dev/peps/pep-0008/) (use `black` formatter)
- Add tests for new features
- Update `requirements.txt` for deps
- Document changes in README

**Issues?** Open a ticket with logs/screenshots from `Screenshots/`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- **Neo4j Team**: Powerful graph database
- **Pinecone**: Vector search infrastructure
- **OpenAI**: Cutting-edge embeddings & generation
- **Pyvis**: Beautiful graph visualizations
- **Dataset**: Curated Vietnam travel locations (public domain sources)

---

**Built with ❤️ for travel enthusiasts!**  
**Questions?** [Open an issue](https://github.com/Nikhil18207/Blue-Enigma-Task/issues).

---

<div align="center">
  <img src="https://img.shields.io/github/stars/Nikhil18207/Blue-Enigma-Task?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/Nikhil18207/Blue-Enigma-Task?style=social" alt="GitHub forks">
</div>
