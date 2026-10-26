export type HealthResponse = {
  status: string;
  version: string;
  database_ok: boolean;
  inference_mode: string;
  base_model: string;
  adapter_loaded: boolean;
  rag_enabled: boolean;
  embedding_model: string;
  disclaimer: string;
  detail?: string | null;
};

export type SourceCitation = {
  document_name: string;
  page_number: number;
  chunk_id: string;
  score: number;
  excerpt: string;
};

export type ChatResponse = {
  conversation_id: string;
  message_id: string;
  answer: string;
  model_mode: string;
  latency_ms: number;
  sources: SourceCitation[];
  safety_flags: Record<string, unknown>;
  disclaimer: string;
  sufficient: boolean;
};

export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type MessageOut = {
  id: string;
  role: string;
  content: string;
  model_mode?: string | null;
  latency_ms?: number | null;
  sources?: SourceCitation[] | null;
  safety_flags?: Record<string, unknown> | null;
  created_at: string;
};

export type ConversationDetail = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: MessageOut[];
};

export type DocumentOut = {
  id: string;
  filename: string;
  original_name: string;
  page_count: number;
  chunk_count: number;
  status: string;
  error_message?: string | null;
  created_at: string;
};

export type DatasetStatsOut = {
  source_name: string;
  total_raw: number;
  valid: number;
  duplicates_removed: number;
  train_count: number;
  val_count: number;
  test_count: number;
  stats_json?: Record<string, unknown> | null;
  created_at?: string | null;
  processed_available: boolean;
};

export type TrainingRunOut = {
  id: string;
  base_model: string;
  adapter_path?: string | null;
  hardware_profile: string;
  status: string;
  config?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  error_message?: string | null;
  created_at: string;
  finished_at?: string | null;
};

export type EvalRunOut = {
  id: string;
  name: string;
  model_mode: string;
  sample_size: number;
  metrics: Record<string, number | string>;
  comparison?: Record<string, unknown> | null;
  notes?: string | null;
  created_at: string;
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
