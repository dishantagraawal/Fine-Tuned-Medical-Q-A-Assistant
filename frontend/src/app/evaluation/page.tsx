"use client";

import { useEffect, useState } from "react";
import { fetchEvalRuns } from "@/lib/api";
import type { EvalRunOut } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function EvaluationPage() {
  const [runs, setRuns] = useState<EvalRunOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEvalRuns()
      .then(setRuns)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load evaluation"),
      );
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Evaluation dashboard</CardTitle>
        <CardDescription>
          Shows only metrics from actual evaluation runs — never fabricated scores.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
{`cd backend && source .venv/bin/activate
python -m ml evaluate --max-samples 50 --compare-base`}
        </pre>
        {error && <p className="text-sm text-rose-600">{error}</p>}
        {runs.length === 0 && !error && (
          <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
            No evaluation runs yet. Run{" "}
            <code className="rounded bg-white px-1">python -m ml evaluate</code>{" "}
            after preprocessing (and optionally training) to populate this dashboard.
          </div>
        )}
        {runs.map((run) => (
          <div
            key={run.id}
            className="rounded-xl border border-slate-200 bg-white p-4"
          >
            <div className="flex flex-wrap items-center gap-2">
              <p className="font-semibold text-slate-900">{run.name}</p>
              <Badge>{run.model_mode}</Badge>
              <Badge>{run.sample_size} samples</Badge>
              <span className="text-xs text-slate-500">
                {new Date(run.created_at).toLocaleString()}
              </span>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(run.metrics || {}).map(([key, value]) => (
                <div
                  key={key}
                  className="rounded-lg border border-slate-100 bg-slate-50 p-3"
                >
                  <p className="text-xs uppercase text-slate-500">{key}</p>
                  <p className="text-lg font-semibold text-slate-900">
                    {typeof value === "number" ? value.toLocaleString() : String(value)}
                  </p>
                </div>
              ))}
            </div>
            {run.comparison && (
              <pre className="mt-3 max-h-40 overflow-auto rounded bg-slate-50 p-2 text-xs">
                {JSON.stringify(run.comparison, null, 2)}
              </pre>
            )}
            {run.notes && (
              <p className="mt-2 text-xs text-slate-500">{run.notes}</p>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
