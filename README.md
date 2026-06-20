# OmniTrust-RAG 🛡️ — Next-Generation Multi-Agent Evidence Verification

OmniTrust-RAG is a production-ready, full-stack Retrieval-Augmented Generation (RAG) system with a real-time evidence verification engine. It fuses concepts from three advanced research architectures:
1. **Adaptive Evidence RAG:** Factual retrieval that evaluates evidence across source independence (via NLI/TF-IDF cosine similarity clustering) and information utility (novelty vs. contradiction).
2. **Pluto Multi-Agent System:** Factual synthesis driven by an active Planner-Strategist agent council, with automatic Critic auditing and Synthesizer claim formatting.
3. **FamilyAttn Verification:** A local 4-head attention verifier analyzing divergence across Semantic, Named Entity, Temporal, and Context representations using Jensen-Shannon (JS) divergence.

Featuring a **world-class dark mode dashboard (Obsidian Neo-Glow)** built with React and Vite, the UI offers a dynamic pipeline progress visualizer, interactive consensus heatmaps, and structured agent terminal traces, free of generic AI styling tropes.

---

## 🏗️ Architecture Flow

```
                         [ User Query ]
                               │
                    [ Planner / Strategist ]
                               │
                      [ Factual Retrieval ]
                               │
                  [ Scorers: Novelty, Indep ]
                               │
                 [ Family Attention Verifier ]
                               │
                      [ Critic / Synthesize ]
                               │
                        [ Final Answer ]
```

---

## 📂 Project Structure

```
omnitrust-rag/
├── README.md                      # Core documentation (this file)
├── .gitignore                     # Root Git exclusions (build artifacts, env, node_modules)
├── backend/                       # Python FastAPI Backend
│   ├── main.py                    # Entrypoint (endpoints, CORS, SPA static file serving)
│   ├── requirements.txt           # Python backend dependencies
│   ├── test_pipeline.py           # Integration test script
│   ├── evaluate.py                # 10-question evaluation benchmark
│   └── omnitrust/
│       ├── bus.py                 # Thread-safe agent MessageBus
│       ├── config.py              # Global settings & API credentials
│       ├── agents.py              # LLM agent definitions (Planner, Strategist, Critic, Synthesizer)
│       ├── retriever.py           # Hybrid Wikipedia/Corpus retriever + NVIDIA NIM Reranking
│       ├── scorer.py              # Source Independence & Utility scorers
│       ├── family_verifier.py     # 4-head Family Attention verifier (JS Divergence)
│       └── pipeline.py            # 10-step orchestrator pipeline
└── frontend/                      # Vite + React Frontend
    ├── index.html                 # App shell (Plus Jakarta Sans & JetBrains Mono)
    ├── package.json               # Node dependencies
    ├── vite.config.js             # Vite configuration with API reverse proxy
    └── src/
        ├── main.jsx               # React main hook
        ├── api.js                 # API endpoints bindings
        ├── index.css              # Obsidian Neo-Glow style sheet
        └── App.jsx                # Responsive telemetry dashboard
```

---

## ⚡ 10-Step Orchestration Pipeline

When you submit a query, the backend executes the following loop:
1. **Planner Agent:** Formulates 3-5 search queries via Groq (`llama-3.1-8b-instant`).
2. **Strategist Agent:** Evaluates and optimizes the queries via Groq (`llama-3.3-70b-versatile`).
3. **Hybrid Retriever:** Performs parallel retrieval from Wikipedia and the in-memory document store.
4. **NVIDIA NIM Reranking:** Calls NVIDIA's QA-Mistral-4B Cloud Rerank endpoint to rank retrieved documents.
5. **Source Independence Scorer:** Clusters documents via TF-IDF cosine similarity to identify redundancy.
6. **Information Utility Scorer:** Scores items based on novelty, length, and contradiction detection.
7. **Document Filtering:** Removes redundant duplicates and low-utility source documents.
8. **Family Attention Verifier:** Runs a local 4-head divergence check to gauge cross-source consensus.
9. **Critic Agent:** Cross-checks and fact-audits all claims against raw evidence.
10. **Synthesizer Agent:** Generates the final cited response, rendering it live in the UI.

---

## 🚀 Quick Start (Development Mode)

### Prerequisites
* Python 3.9+
* Node.js 18+
* Groq API Key and NVIDIA API Key

### 1. Set Up Environment Variables
Create a `.env` file in the `backend/` directory or export them in your terminal:
```bash
export GROQ_API_KEY="your-groq-api-key"
export NVIDIA_API_KEY="your-nvidia-nim-api-key"
```

### 2. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
*Backend runs on `http://localhost:8000`.*

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
*Frontend runs on `http://localhost:5173` (proxies `/api` automatically).*

---

## 📦 Production Deployment & Scaling

For mass production and deployment, OmniTrust-RAG supports **Single-Container Unified Serving**. By compiling the React frontend into static assets, FastAPI serves the complete dashboard and API from a single server.

### 1. Build the Frontend Assets
Compile the static HTML/JS/CSS assets:
```bash
cd frontend
npm run build
```
This writes the production build to `frontend/dist/`. The FastAPI backend in [backend/main.py](file:///d:/res_core_kimi/omnitrust-rag/backend/main.py) will automatically detect this directory and serve the dashboard at `http://localhost:8000`.

### 2. Docker Containerization
To deploy as a Docker container, use the following configurations:

#### Backend Dockerfile (`backend/Dockerfile`):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Multi-Stage Dockerfile (Root `Dockerfile`):
A single multi-stage build that bundles frontend files inside the FastAPI python server:
```dockerfile
# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve via FastAPI python app
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-builder /frontend/dist /frontend/dist
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Run with Docker Compose (`docker-compose.yml`):
```yaml
version: '3.8'
services:
  omnitrust-rag:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    restart: unless-stopped
```

---

## 🛡️ Production Scaling Best Practices

1. **Groq Rate-Limiting:** Groq's free-tier `llama-3.1-8b-instant` has strict Tokens-Per-Minute (TPM) limits. For production:
   - Implement query and document summarization caching (Redis).
   - Use enterprise Groq tiers or switch to OpenAI/Anthropic model fallbacks in `backend/omnitrust/agents.py`.
2. **Asynchronous Task Queue:** In mass use cases, replace synchronous inline pipeline runs with Celery / Redis background tasks, notifying the frontend via WebSockets.
3. **Static Corpus Storage:** Replace the in-memory corpus database with a persistent Vector database (e.g., pgvector, Qdrant, Chroma) for scaling document store sizes.

---

## 🐙 Push to GitHub

To push this project to your GitHub repository:

1. Initialize a Git repository at the root of the project:
   ```bash
   git init
   ```
2. Add all project files:
   ```bash
   git add .
   ```
3. Commit the code:
   ```bash
   git commit -m "feat: Fused Multi-Agent RAG with FamilyAttn verifiers and Obsidian Neo-Glow UI"
   ```
4. Link to your remote GitHub repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```
