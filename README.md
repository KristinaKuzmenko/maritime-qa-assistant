# Maritime QA Assistant

**Advanced RAG System for Technical Maritime Documentation**

Maritime QA Assistant is an intelligent question-answering system designed for technical maritime documents. It uses an **agentic LangGraph workflow** that dynamically selects appropriate tools and data sources to provide accurate answers with citations.

---

## 🚀 Key Features

- **🤖 Agentic RAG**: LLM-based agent autonomously decides which tools to use (Qdrant vector search, Neo4j graph traversal, entity recognition)
- **📊 Multi-Modal Context**: Processes text chunks, tables, schemas (P&ID diagrams), and entities
- **🧠 Smart Entity Detection**: Automatic equipment code recognition with graph-based navigation
- **🔍 Intent Classification**: Routes queries to appropriate data sources (text/table/schema/mixed)
- **☁️ Cloud-Native**: AWS S3 storage, Neo4j knowledge graph, Qdrant vector database
- **🔐 Security**: Prompt injection protection, JWT authentication, rate limiting
- **📈 GitOps Deployment**: Automated CI/CD with GitHub Actions, ArgoCD, and AWS EKS

---

## 🏗️ Architecture

```
User Query → Streamlit UI → FastAPI Backend → LangGraph Workflow
                                                    ↓
                                    ┌──────────────┼──────────────┐
                                    ↓              ↓              ↓
                                 Qdrant         Neo4j          AWS S3
                              (Vector DB)    (Graph DB)     (Documents)
```

**Tech Stack:**
- **Frontend:** Streamlit
- **Backend:** FastAPI + LangGraph
- **Databases:** Qdrant (vectors), Neo4j (knowledge graph)
- **Storage:** AWS S3
- **LLMs:** OpenAI GPT-4o, Groq Llama, Cerebras
- **Infrastructure:** Terraform, Kubernetes (EKS), ArgoCD

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Complete deployment guide (local, Docker, AWS EKS) |
| **[TESTING.md](TESTING.md)** | Testing guide (unit tests, benchmarks, CI/CD) |
| **[WORKFLOW.md](WORKFLOW.md)** | Agentic RAG workflow architecture and logic |
| **[INGESTION_PIPELINE.md](INGESTION_PIPELINE.md)** | Document processing pipeline (YOLO, OCR, entity extraction) |
| **[EVALUATION.md](EVALUATION.md)** | RAG evaluation metrics and benchmarking |
| **[PROMPT_INJECTION_PROTECTION.md](PROMPT_INJECTION_PROTECTION.md)** | Security and prompt injection prevention |
| **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** | Performance benchmarking (load tests, real API tests) |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Neo4j AuraDB** account (free tier: [neo4j.com/aura](https://neo4j.com/aura))
- **OpenAI API** key ([platform.openai.com](https://platform.openai.com))
- **Qdrant Cloud** (optional, free tier: [cloud.qdrant.io](https://cloud.qdrant.io)) OR use local Qdrant
- **AWS Account** (for S3 storage)

### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/KristinaKuzmenko/maritime-qa-assistant.git
cd maritime-qa-assistant

# Copy environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required variables in `.env`:**
```bash
# Neo4j (Required)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# OpenAI (Required for embeddings)
OPENAI_API_KEY=sk-proj-xxxxx

# Qdrant (use local OR cloud)
QDRANT_HOST=localhost  # For local Qdrant
QDRANT_PORT=6333
# OR for Qdrant Cloud:
# QDRANT_HOST=xxxxx.cloud.qdrant.io
# QDRANT_API_KEY=your-api-key
# QDRANT_PORT=6333

# AWS S3 (Required)
S3_BUCKET_NAME=maritime-qa-files
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# LLM Providers (Optional)
GROQ_API_KEY=your-groq-key  # Fast inference
CEREBRAS_API_KEY=your-cerebras-key

# Authentication
AUTH_SECRET_KEY=your-32-character-secret-key
```

### 2. Run Locally (Without Docker)

**Simplest way to run for development:**

```bash
# Install Python dependencies
pip install -r requirements.txt

# Terminal 1: Start backend (FastAPI)
python backend/main.py
# Backend will start on http://localhost:8000

# Terminal 2: Start frontend (Streamlit)
streamlit run frontend/app.py
# Frontend will open on http://localhost:8501
```

**Access:**
- **Frontend UI:** http://localhost:8501 (opens automatically in browser)
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Logs:** Displayed in terminal (real-time)

**Benefits:**
- ✅ Fast startup (no Docker build)
- ✅ Real-time logs in terminal
- ✅ Easy debugging with breakpoints
- ✅ Hot reload on code changes

**Note:** Requires local Qdrant or Qdrant Cloud (set `QDRANT_HOST` in `.env`)

### 3. Run with Docker Compose (MVP - Recommended)

**Simple production-like setup with Qdrant Cloud:**

```bash
# Start application (backend + frontend)
docker-compose -f docker-compose.mvp.yml up -d

# Check logs
docker-compose -f docker-compose.mvp.yml logs -f

# Stop services
docker-compose -f docker-compose.mvp.yml down
```

**Access:**
- **Frontend UI:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

**First-time setup:**
```bash
# Create admin user (after containers are running)
docker-compose -f docker-compose.mvp.yml exec maritime-app-mvp python create_first_user.py
# Default credentials: admin / admin (change after first login!)
```

### 4. Run with Docker Compose (Full Development)

**Includes local Qdrant instance:**

```bash
# Start all services (app + local Qdrant)
docker-compose up -d

# Access Qdrant dashboard: http://localhost:6333/dashboard
```

### 5. Upload Documents

1. Open frontend at http://localhost:8501
2. Go to **"📄 Upload Documents"** page
3. Upload PDF files (maritime technical documentation)
4. Wait for processing (YOLO layout detection, entity extraction, vector indexing)
5. Go to **"💬 Chat"** page and ask questions!

**Example questions:**
- "What are the main functional sections?"
- "Show me the schema for fuel system"
- "What is the specification for pump PU3?"
- "List all tables related to electrical parameters"

---

## 🧪 Testing

### Run Unit Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest backend/tests/ -v

# Run with coverage report
pytest backend/tests/ --cov=backend --cov-report=html
# Open htmlcov/index.html
```

### Run Benchmarks

```bash
# Load testing (local, free)
./benchmark.sh load

# Real API testing (cloud, expensive - check costs first!)
./benchmark.sh real-quick  # Quick smoke test (~$0.0001)
./benchmark.sh real        # Full benchmarks (~$1-10)
```

**See [TESTING.md](TESTING.md) for detailed testing guide.**

---

## 🚀 Deployment to Production

### Quick Deployment Steps

1. **Create AWS Infrastructure:**
   ```bash
   cd aws/infrastructure
   terraform init
   terraform apply -auto-approve
   ```

2. **Setup GitHub Actions OIDC:**
   ```bash
   cd aws/github-actions
   terraform init
   terraform apply -auto-approve
   # Copy role ARN to GitHub Secrets
   ```

3. **Configure GitHub Secrets:**
   - Go to `https://github.com/YOUR_USERNAME/maritime-qa-assistant/settings/secrets/actions`
   - Add: `AWS_ROLE_TO_ASSUME`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPOSITORY`
   - **DO NOT** add AWS credentials (uses OIDC!)

4. **Deploy with ArgoCD:**
   ```bash
   cd aws/argocd
   terraform init
   terraform apply -auto-approve
   kubectl apply -f manifests/maritime-qa-app.yaml
   ```

5. **Push to GitHub:**
   ```bash
   git push origin main
   # GitHub Actions builds Docker image → ECR
   # ArgoCD deploys to EKS automatically
   ```

**Full deployment guide:** See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for step-by-step instructions including:
- AWS EKS cluster setup
- S3 buckets and IAM roles
- External Secrets Operator
- AWS Load Balancer Controller
- Monitoring with Prometheus/Grafana
- Log management

---

## 📊 Evaluation

Evaluate RAG system performance using RAGAS metrics:

```bash
# Install evaluation dependencies
pip install -r requirements-eval.txt

# Run evaluation
python backend/evaluate_rag.py evaluation.json results.json

# View results in results.json and evaluation_plots/
```

**Metrics tracked:**
- **Faithfulness** (answer grounded in context)
- **Answer Relevancy** (addresses the question)
- **Context Precision** (retrieved context quality)
- **Schema/Table Inclusion F1** (correct resource usage)
- **Citation Accuracy** (source attribution)
- **Tool Usage Analysis** (agent decision quality)
- **Latency** (response time)

**See [EVALUATION.md](EVALUATION.md) for detailed evaluation guide.**

---

## 🔐 Security

Maritime QA Assistant implements multi-layer security:

- ✅ **Prompt Injection Protection**: Detects and blocks malicious prompts
- ✅ **JWT Authentication**: Token-based user authentication
- ✅ **Rate Limiting**: Prevents API abuse
- ✅ **OIDC for GitHub Actions**: No AWS credentials in GitHub
- ✅ **IRSA for Kubernetes**: No secrets in Pod specs
- ✅ **Input Validation**: Query sanitization and length limits

**See [PROMPT_INJECTION_PROTECTION.md](PROMPT_INJECTION_PROTECTION.md) for security details.**

---

## 🏗️ Project Structure

```
maritime-qa-assistant/
├── backend/                    # FastAPI backend
│   ├── core/                   # Core services (Neo4j, Qdrant, S3)
│   ├── services/              # Business logic (document processor, workflow)
│   ├── routes/                # API endpoints
│   ├── tests/                 # Unit & integration tests
│   └── main.py                # FastAPI app entry point
├── frontend/                   # Streamlit frontend
│   ├── pages/                 # UI pages (chat, upload, search)
│   └── app.py                 # Streamlit app entry point
├── aws/                       # Terraform infrastructure
│   ├── infrastructure/        # VPC, EKS, ECR, IAM
│   ├── github-actions/        # OIDC setup for GitHub
│   ├── argocd/               # GitOps deployment
│   └── k8s/                  # Kubernetes manifests
├── k8s/                       # Application K8s manifests
│   └── base/                 # Deployment, Service, ConfigMap
├── .github/workflows/         # CI/CD pipelines
├── docker-compose.yml         # Development setup (with local Qdrant)
├── docker-compose.mvp.yml     # Production-like setup (Qdrant Cloud)
├── Dockerfile                 # Multi-stage build (test + production)
└── requirements.txt           # Python dependencies
```

---

## 📝 Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j+s://xxxxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `your-password` |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) | `sk-proj-xxxxx` |
| `S3_BUCKET_NAME` | AWS S3 bucket for documents | `maritime-qa-files` |
| `S3_REGION` | AWS region | `us-east-1` |
| `AUTH_SECRET_KEY` | JWT secret (32+ chars) | `your-secret-key` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `QDRANT_API_KEY` | Qdrant Cloud API key | `None` |
| `GROQ_API_KEY` | Groq API key (fast inference) | `None` |
| `CEREBRAS_API_KEY` | Cerebras API key | `None` |
| `LLM_PROVIDER` | LLM provider | `openai` |
| `LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

**Full configuration reference:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#configuration-reference)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Testing requirements:**
- All tests must pass (`pytest backend/tests/ -v`)
- Code coverage > 80% for new code
- Follow existing code style

---

## 📄 License

This project is licensed under the MIT License.

---

## 🆘 Troubleshooting

### Common Issues

**1. Cannot connect to Neo4j**
```
Error: ServiceUnavailable: WebSocket connection failure
```
Solution: Check Neo4j URI and credentials in `.env`. Ensure Neo4j Aura is running.

**2. Qdrant connection refused**
```
Error: Connection refused [Errno 111]
```
Solution: 
- For local Qdrant: `docker-compose up -d` to start Qdrant container
- For Qdrant Cloud: Check `QDRANT_HOST` and `QDRANT_API_KEY` in `.env`

**3. S3 access denied**
```
Error: botocore.exceptions.ClientError: Access Denied
```
Solution: Check AWS credentials in `.env`. Verify IAM permissions for S3 bucket.

**4. Docker build fails**
```
Error: ModuleNotFoundError: No module named 'XXX'
```
Solution: Rebuild with `--no-cache`: `docker-compose build --no-cache`

**5. Frontend not loading**
```
Error: Connection refused at http://localhost:8501
```
Solution: 
- Check logs: `docker-compose logs frontend`
- Ensure backend is running first
- Wait 30-60 seconds for services to start

### Get Help

- **Documentation:** See files in this repository
- **Issues:** Open GitHub issue with logs and error messages
- **Logs:** `docker-compose logs -f` for debugging

---

## 📬 Contact

- **GitHub:** [@KristinaKuzmenko](https://github.com/KristinaKuzmenko)
- **Repository:** [maritime-qa-assistant](https://github.com/KristinaKuzmenko/maritime-qa-assistant)

---

## 🙏 Acknowledgments

- **LangChain/LangGraph** - Agentic workflow framework
- **Qdrant** - Vector database
- **Neo4j** - Graph database
- **OpenAI** - LLM and embeddings
- **Streamlit** - UI framework
- **FastAPI** - Backend framework
