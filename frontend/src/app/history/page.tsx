"use client";

import { useEffect, useState } from "react";
import { fetchConversation, fetchConversations } from "@/lib/api";
import type { ConversationDetail, ConversationSummary } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function HistoryPage() {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [selected, setSelected] = useState<ConversationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConversations()
      .then(setItems)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load history"),
      );
  }, []);

  async function openConversation(id: string) {
    try {
      setSelected(await fetchConversation(id));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversation");
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Conversation history</CardTitle>
          <CardDescription>Persisted in PostgreSQL</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {error && <p className="text-sm text-rose-600">{error}</p>}
          {items.length === 0 && !error && (
            <p className="text-sm text-slate-500">No conversations yet.</p>
          )}
          {items.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => openConversation(c.id)}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left hover:border-teal-400"
            >
              <p className="truncate text-sm font-medium text-slate-800">
                {c.title}
              </p>
              <p className="text-xs text-slate-500">
                {c.message_count} messages ·{" "}
                {new Date(c.updated_at).toLocaleString()}
              </p>
            </button>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{selected?.title ?? "Select a conversation"}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {!selected && (
            <p className="text-sm text-slate-500">
              Choose a conversation to inspect messages, latency, and sources.
            </p>
          )}
          {selected?.messages.map((m) => (
            <div
              key={m.id}
              className="rounded-lg border border-slate-200 bg-white p-3 text-sm"
            >
              <div className="mb-1 flex items-center gap-2">
                <Badge>{m.role}</Badge>
                {m.model_mode && <Badge>{m.model_mode}</Badge>}
                {m.latency_ms != null && (
                  <span className="text-xs text-slate-500">
                    {m.latency_ms.toFixed(0)} ms
                  </span>
                )}
              </div>
              <p className="whitespace-pre-wrap text-slate-800">{m.content}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
