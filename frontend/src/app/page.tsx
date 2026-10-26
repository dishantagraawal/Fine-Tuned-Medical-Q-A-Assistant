"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { askChat, fetchHealth, getApiBaseUrl } from "@/lib/api";
import type { ChatResponse, HealthResponse, SourceCitation } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

type Turn = {
  role: "user" | "assistant";
  content: string;
  meta?: ChatResponse;
};

const EXAMPLES = [
  "What causes benign paroxysmal positional vertigo?",
  "Explain hypertension in simple terms.",
  "What are common side effects of beta blockers?",
];

export default function ChatPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [useRag, setUseRag] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchHealth();
        if (!cancelled) {
          setHealth(data);
          setHealthError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setHealthError(
            err instanceof Error
              ? err.message
              : `Cannot reach API at ${getApiBaseUrl()}`,
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = message.trim();
    if (!text || pending) return;
    setError(null);
    setPending(true);
    setTurns((prev) => [...prev, { role: "user", content: text }]);
    setMessage("");
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await askChat(text, {
        conversationId,
        useRag,
        signal: controller.signal,
      });
      setConversationId(res.conversation_id);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, meta: res },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed");
      setTurns((prev) => prev.slice(0, -1));
      setMessage(text);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <Card className="min-h-[70vh]">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle>Medical chatbot</CardTitle>
              <CardDescription>
                Ask clinical or patient-style questions in plain language.
              </CardDescription>
            </div>
            <Badge
              className={
                health?.status === "ok"
                  ? "border-teal-200 bg-teal-50 text-teal-800"
                  : "border-amber-200 bg-amber-50 text-amber-800"
              }
            >
              {health
                ? `${health.status} · ${health.inference_mode}${
                    health.adapter_loaded ? " · adapter" : ""
                  }`
                : healthError
                  ? "API offline"
                  : "Checking…"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="flex h-[calc(70vh-5rem)] flex-col gap-4">
          <div className="flex-1 space-y-3 overflow-y-auto rounded-lg bg-slate-50 p-4">
            {turns.length === 0 && (
              <div className="space-y-3 text-sm text-slate-600">
                <p>Try an example:</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs hover:border-teal-400"
                      onClick={() => setMessage(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((turn, idx) => (
              <div
                key={`${turn.role}-${idx}`}
                className={
                  turn.role === "user"
                    ? "ml-8 rounded-xl bg-teal-700 px-4 py-3 text-sm text-white"
                    : "mr-8 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800"
                }
              >
                <p className="whitespace-pre-wrap">{turn.content}</p>
                {turn.meta && (
                  <div className="mt-3 space-y-2 border-t border-slate-100 pt-2 text-xs text-slate-500">
                    <p>
                      Mode: {turn.meta.model_mode} · Latency:{" "}
                      {turn.meta.latency_ms.toFixed(0)} ms
                    </p>
                    {turn.meta.sources?.length > 0 && (
                      <SourcesList sources={turn.meta.sources} />
                    )}
                    <p className="italic">{turn.meta.disclaimer}</p>
                  </div>
                )}
              </div>
            ))}
            {pending && (
              <p className="text-sm text-slate-500">Generating response…</p>
            )}
          </div>

          {error && (
            <p className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
              {error}
            </p>
          )}

          <form onSubmit={onSubmit} className="space-y-3">
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
              />
              Use RAG (retrieve from uploaded medical PDFs)
            </label>
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Describe symptoms or ask a medical education question…"
              disabled={pending}
            />
            <div className="flex justify-end">
              <Button type="submit" disabled={pending || !message.trim()}>
                Send
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Safety</CardTitle>
          <CardDescription>Always shown with answers</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-slate-600">
          <p>
            Responses are educational and not a substitute for professional
            medical advice, diagnosis, or treatment.
          </p>
          <p>
            Emergency-like language triggers guidance to seek emergency care
            immediately.
          </p>
          <p className="text-xs text-slate-500">
            API: {getApiBaseUrl()}
            {health?.base_model ? ` · ${health.base_model}` : null}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function SourcesList({ sources }: { sources: SourceCitation[] }) {
  return (
    <ul className="space-y-1">
      {sources.map((s) => (
        <li key={s.chunk_id}>
          <span className="font-medium text-slate-700">
            {s.document_name}
          </span>{" "}
          p.{s.page_number} (score {s.score.toFixed(3)})
        </li>
      ))}
    </ul>
  );
}
