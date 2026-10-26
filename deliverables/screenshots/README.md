# Screenshots — model outputs and evaluation results

Captured from the running MedQA Assistant UI (http://localhost:3000) against the live FastAPI backend.

| File | Description |
|------|-------------|
| [01-chat-hypertension-output.png](01-chat-hypertension-output.png) | Chat model output for “Explain hypertension in simple terms” (mock inference mode + disclaimer) |
| [02-evaluation-results.png](02-evaluation-results.png) | Evaluation dashboard with real offline metrics (ROUGE-L, latency, token overlap) |
| [03-dataset-preprocess-stats.png](03-dataset-preprocess-stats.png) | Dataset management: HealthCareMagic preprocess counts and 80/10/10 splits |
| [04-chat-emergency-safety.png](04-chat-emergency-safety.png) | Emergency safety response for chest pain / breathing difficulty |
| [05-chat-rag-with-citation.png](05-chat-rag-with-citation.png) | Chat UI with RAG enabled (companion JSON in `../samples/response-rag-hypertension.json`) |
| [06-documents-library.png](06-documents-library.png) | Medical document library with indexed PDF (`sample_medical_note.pdf`) |
| [07-conversation-history.png](07-conversation-history.png) | Conversation history persisted in PostgreSQL |
| [08-history-conversation-detail.png](08-history-conversation-detail.png) | History detail: safety + mock/RAG answer with modes and latency |

**Note:** Screenshots showing `mock` inference were taken with `MOCK_INFERENCE=true` (no GPU weight download). After fine-tuning, set `MOCK_INFERENCE=false`, reload the adapter, and re-run `python -m ml evaluate` for trained-model screenshots.
