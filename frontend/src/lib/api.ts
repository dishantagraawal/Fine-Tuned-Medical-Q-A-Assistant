import type {
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  DatasetStatsOut,
  DocumentOut,
  EvalRunOut,
  HealthResponse,
  TrainingRunOut,
} from "@/lib/types";
import { ApiError } from "@/lib/types";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  return (raw || DEFAULT_API_BASE).replace(/\/$/, "");
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((item) =>
          typeof item === "object" && item && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : JSON.stringify(item),
        )
        .join("; ");
    }
  } catch {
    /* ignore */
  }
  return response.statusText || `Request failed (${response.status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { method: "GET", signal });
}

export function askChat(
  message: string,
  opts?: { conversationId?: string; useRag?: boolean; signal?: AbortSignal },
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: opts?.conversationId ?? null,
      use_rag: opts?.useRag ?? false,
    }),
    signal: opts?.signal,
  });
}

export function fetchConversations(): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/api/conversations");
}

export function fetchConversation(id: string): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/api/conversations/${id}`);
}

export function fetchDocuments(): Promise<DocumentOut[]> {
  return request<DocumentOut[]>("/api/documents");
}

export async function uploadDocument(file: File): Promise<DocumentOut> {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentOut>("/api/documents/upload", {
    method: "POST",
    body: form,
  });
}

export function deleteDocument(id: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/api/documents/${id}`, { method: "DELETE" });
}

export function fetchDatasetStats(): Promise<DatasetStatsOut> {
  return request<DatasetStatsOut>("/api/dataset/stats");
}

export function fetchTrainingRuns(): Promise<TrainingRunOut[]> {
  return request<TrainingRunOut[]>("/api/training/runs");
}

export function registerTrainingRun(body: {
  max_samples?: number;
  epochs?: number;
  hardware_profile?: string;
}): Promise<TrainingRunOut> {
  return request<TrainingRunOut>("/api/training/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchEvalRuns(): Promise<EvalRunOut[]> {
  return request<EvalRunOut[]>("/api/evaluation/runs");
}
