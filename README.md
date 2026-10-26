# Fine-Tuned Medical Q&A Assistant

Full-stack educational medical Q&A system with:

- **Frontend:** Next.js, TypeScript, Tailwind CSS, shadcn-style UI
- **Backend:** Python FastAPI
- **AI/ML:** PyTorch, Hugging Face Transformers, TRL, PEFT (LoRA/QLoRA)
- **Database:** PostgreSQL + pgvector (chat history + document RAG)
- **Deployment:** Docker Compose

Primary dataset: `HealthCareMagic-100k.json` (zip archive containing 112k+ `instruction` / `input` / `output` records).

> **Disclaimer:** This assistant provides educational information only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. If you may be having a medical emergency, call local emergency services immediately.

## Architecture

```text
Next.js dashboard  →  FastAPI  →  PostgreSQL/pgvector
                         ├─ Safety checks
                         ├─ Fine-tuned / base HF inference (or mock)
                         └─ Optional RAG over uploaded PDFs
Dataset ZIP → preprocess → train/val/test JSONL → LoRA/QLoRA (CLI)
```

Patterns for PDF chunking, disclaimers, and FastAPI CORS were adapted from the local `ai-driven-healthcare-app` reference project (FAISS/Ollama there; pgvector + fine-tuning here).

## Prerequisites

- Docker + Docker Compose **or**
- Node.js 20+, Python 3.9+, PostgreSQL 16 with pgvector

## Quick start (Docker)

```bash
cd "/Users/dishantagrawal/Desktop/Fine-Tuned Medical Q&A Assistant"

cp .env.example .env
cp frontend/.env.example frontend/.env.local

docker compose up --build
```

- UI: http://localhost:3000  
- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/api/health  
- Postgres (host): `localhost:5433` → container `5432` (avoids clashing with a local Postgres on 5432)

Docker API defaults to `MOCK_INFERENCE=true` so the stack starts without downloading model weights.

## Local development

### 1. Database

```bash
docker compose up -d db
# Host port 5433 → container 5432
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env
# ensure MOCK_INFERENCE=true for first run without GPU weights

export PYTHONPATH=.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

## Dataset preprocessing

The file `HealthCareMagic-100k.json` is a **ZIP**. Preprocess extracts it, maps `input`→question / `output`→answer, validates, deduplicates, and splits **80/10/10** with hash-based partitioning (no question leakage across splits).

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=.
python -m ml preprocess
```

Outputs:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/stats.json`

## Fine-tuning (do not run accidentally)

Training is **manual** and can be expensive. Configure `backend/configs/train.yaml`:

| Profile | Notes |
|--------|--------|
| `cpu` / `mps` | LoRA, smaller `max_samples` (default 2000) |
| `cuda` | Set `use_qlora: true` for 4-bit QLoRA |

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=.
# edit configs/train.yaml first
python -m ml train --config configs/train.yaml
```

Adapter + metrics land in `data/adapters/latest/` and `training_metrics.json`.

## Evaluation (real metrics only)

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=.
python -m ml evaluate --max-samples 50 --compare-base
```

Results write to `data/processed/eval_results.json` and appear on the Evaluation page. The UI never invents scores.

## RAG (optional)

1. Open **Documents** in the UI and upload medical PDFs  
2. In **Chat**, enable “Use RAG”  
3. Answers include document/page citations when retrieval passes the similarity threshold  

## Tests

```bash
# Backend
cd backend && source .venv/bin/activate
export PYTHONPATH=.
export MOCK_INFERENCE=true
pytest -q

# Frontend unit tests (Node built-in test runner)
cd frontend
node --import tsx --test src/lib/api.test.ts
# or, if tsx is unavailable:
npx --yes tsx --test src/lib/api.test.ts
```

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | DB + inference status |
| POST | `/api/chat` | Chat with optional RAG |
| GET | `/api/conversations` | History list |
| GET/POST | `/api/documents` / upload | Document library |
| GET | `/api/dataset/stats` | Preprocess stats |
| GET/POST | `/api/training/runs` | Training run registry |
| GET | `/api/evaluation/runs` | Stored eval metrics |

## Limitations

- Full HealthCareMagic (~112k) fine-tuning needs a suitable GPU; laptop defaults use a subset.
- Apple Silicon: use `hardware_profile: mps` and `use_qlora: false` (bitsandbytes QLoRA is CUDA-oriented).
- Without downloaded weights, the API runs in **mock inference** mode (clearly labeled).
- iCliniq.json inside the zip is not used by default.
- Educational use only — not a clinical decision-support product.

## Project layout

```text
├── frontend/          Next.js dashboard
├── backend/           FastAPI + ml/preprocess|train|evaluate
├── data/              processed splits, uploads, adapters
├── deliverables/      Project report, screenshots, sample outputs
├── docker-compose.yml
├── HealthCareMagic-100k.json   # zip archive of the dataset
└── README.md
```

## Deliverables (report & screenshots)

- Project report: [`deliverables/01-project-report.md`](deliverables/01-project-report.md)
- Screenshots: [`deliverables/screenshots/`](deliverables/screenshots/)
- Sample API/eval JSON: [`deliverables/samples/`](deliverables/samples/)
