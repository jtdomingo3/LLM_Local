# 🚀 LLM Local Microservice (Standalone Mode)

> **Self-contained, production-ready local LLM microservice** with automated headless engine management.  
> Exposes OpenAI-compatible and task-oriented REST & Server-Sent Events (SSE) streaming APIs on port `8100`.
> 
> **Zero Manual Setup:** Automatically starts and stops the optimized inference engine in the background with CUDA Flash Attention and Q8_0 KV Cache. Simply run `python run.py`.

---

## 💻 Designed & Calibrated For

- **GPU**: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM, Ampere architecture)
- **CPU**: Intel Core i5 12th Gen
- **System Memory**: 24GB DDR5 RAM
- **Storage**: Fast NVMe SSD

---

## ⚡ Key Architectural Optimizations

| Optimization | Configuration / Value | Performance Impact |
|---|---|---|
| **Standalone Engine Lifecycle** | Auto-managed via `engine_manager.py` | 1-command startup/shutdown, no external tray apps needed |
| **Flash Attention** | `OLLAMA_FLASH_ATTENTION=1` | ~25–35% faster prompt evaluation, lower VRAM footprint |
| **KV Cache Quantization** | `OLLAMA_KV_CACHE_TYPE=q8_0` | ~50% KV cache VRAM reduction, enables 4k context within 4GB VRAM |
| **Quantization Scheme** | `Q4_K_M` (imatrix calibrated) | Near-FP16 perplexity with only ~2.2–2.5 GB VRAM usage |
| **Context Window** | `num_ctx = 4096` | Safe headroom for 4GB VRAM avoiding out-of-memory spill |
| **Async Non-blocking I/O** | `AsyncOpenAI` client + FastAPI | High-throughput concurrent request handling |
| **OpenAI Compatibility** | `/v1/chat/completions`, `/v1/models` | Drop-in support for Continue.dev, Cursor, Aider, Chatbox |
| **Streaming Telemetry** | Server-Sent Events (SSE) | Near-zero Time-to-First-Token (TTFT) |

---

## 🧠 Model Tiers

| Tier | Model | Parameters | VRAM Footprint | Speed (RTX 3050) | Primary Purpose |
|---|---|---|---|---|---|
| **Daily Driver** | `phi4-mini` | 3.8B | ~2.5 GB (100% GPU) | ~25 tokens/s | General chat, summarization, Q&A |
| **Fast Coding** | `qwen2.5-coder:3b` | 3.0B | ~2.2 GB (100% GPU) | ~28 tokens/s | Code completion, boilerplate, bugs |
| **Power Coding** | `qwen2.5-coder:7b` | 7.6B | ~4.5 GB (Hybrid GPU/CPU) | ~8 tokens/s | Complex algorithms, deep refactoring |

---

## 📁 Repository Structure

```
c:\Users\Jeff\repo\LLM_Local\
├── .env                          # Local environment configuration
├── .env.example                  # Template environment variables
├── requirements.txt              # Production and test dependencies
├── run.py                        # Single entry point (auto-starts backend)
├── README.md                     # Documentation
├── models/                       # Hard-linked GGUF models on disk
│   ├── phi4-mini.gguf            # 2.5 GB
│   ├── qwen2.5-coder-3b.gguf     # 1.9 GB
│   └── qwen2.5-coder-7b.gguf     # 4.7 GB
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory, CORS, lifespan probes
│   ├── config.py                 # Pydantic Settings & hardware defaults
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── logging_middleware.py # Request ID tracing & latency logging
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic V2 request & response schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py               # /api/chat & /api/chat/stream
│   │   ├── code.py               # /api/code, /stream, /review, /explain
│   │   ├── health.py             # /health/live & /health/ready
│   │   ├── models.py             # /api/models & /api/models/active
│   │   └── openai_compat.py      # /v1/chat/completions, /v1/models (Continue.dev)
│   └── services/
│       ├── __init__.py
│       ├── engine_manager.py     # Headless engine auto-start & cleanup
│       └── llm_service.py        # Async inference, health, and GPU metrics
└── tests/                        # 19 passing unit tests
```

---

## 🛠️ Getting Started

### 1. Configure Ollama & Pull Models (One-Click)

Run the included PowerShell script to configure Windows environment variables for Flash Attention and KV Cache quantization, and pull the calibrated models:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_ollama.ps1
```

### 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the Microservice

```powershell
python run.py
```

*By default, the microservice runs on `http://0.0.0.0:8100`.*

---

## 🔌 API Endpoints Reference

### Interactive Documentation
- **Swagger UI**: [http://localhost:8100/docs](http://localhost:8100/docs)
- **ReDoc**: [http://localhost:8100/redoc](http://localhost:8100/redoc)

---

### 1. Health Checks

#### Liveness Probe (`GET /health/live`)
Confirms the FastAPI process is active.
```bash
curl http://localhost:8100/health/live
```
```json
{
  "status": "alive"
}
```

#### Readiness Probe (`GET /health/ready`)
Verifies connection to Ollama backend, model availability, and GPU status.
```bash
curl http://localhost:8100/health/ready
```
```json
{
  "status": "ready",
  "ollama_connected": true,
  "models_available": ["phi4-mini", "qwen2.5-coder:3b", "qwen2.5-coder:7b"],
  "gpu_detected": true,
  "uptime_seconds": 154.2
}
```

---

### 2. Chat Endpoints

#### Standard Chat (`POST /api/chat`)
```bash
curl -X POST http://localhost:8100/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are three advantages of asynchronous programming in Python?",
    "temperature": 0.7
  }'
```
```json
{
  "model": "phi4-mini",
  "response": "1. Non-blocking I/O operations...\n2. Concurrency with lower memory overhead...\n3. Enhanced scalability for web servers...",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 82,
    "total_tokens": 100
  },
  "latency_ms": 3120.4
}
```

#### Streaming Chat (`POST /api/chat/stream`)
Real-time SSE token stream:
```bash
curl -N -X POST http://localhost:8100/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a short poem about clean code."}'
```
Stream format:
```text
data: {"token": "Clean ", "model": "phi4-mini"}
data: {"token": "lines ", "model": "phi4-mini"}
data: {"token": "flow ", "model": "phi4-mini"}
...
data: {"done": true, "model": "phi4-mini"}
```

---

### 3. Code Generation Endpoints

#### Generate Code (`POST /api/code`)
```bash
curl -X POST http://localhost:8100/api/code \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a FastAPI dependency for validating JWT bearer tokens",
    "language": "python"
  }'
```

#### Review Code (`POST /api/code/review`)
```bash
curl -X POST http://localhost:8100/api/code/review \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def load_config(path):\n    f = open(path)\n    return json.loads(f.read())",
    "language": "python"
  }'
```

#### Explain Code (`POST /api/code/explain`)
```bash
curl -X POST http://localhost:8100/api/code/explain \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "const memoize = fn => { const cache = new Map(); return (...args) => { const k = JSON.stringify(args); return cache.has(k) ? cache.get(k) : (cache.set(k, fn(...args)), cache.get(k)); }; };",
    "language": "javascript"
  }'
```

---

### 4. Models Catalog

#### List Installed Models (`GET /api/models`)
```bash
curl http://localhost:8100/api/models
```

#### View Active Defaults (`GET /api/models/active`)
```bash
curl http://localhost:8100/api/models/active
```

---

## 💻 Client Integration Examples

### Python (`httpx`)
```python
import httpx

# Synchronous generation
with httpx.Client(base_url="http://localhost:8100") as client:
    res = client.post("/api/chat", json={"prompt": "Explain dependency injection"})
    print(res.json()["response"])

# Real-time streaming
with httpx.stream("POST", "http://localhost:8100/api/chat/stream", json={"prompt": "Hello"}) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            print(line[6:], end="", flush=True)
```

### JavaScript / TypeScript (`fetch`)
```javascript
const response = await fetch("http://localhost:8100/api/code", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Write a React hook to detect online/offline status",
    language: "typescript"
  })
});

const data = await response.json();
console.log(data.response);
```

---

## 🧪 Running Tests

Run the full pytest suite:

```powershell
python -m pytest tests/ -v
```
