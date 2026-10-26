"use client";

import { useEffect, useState } from "react";
import { fetchDatasetStats } from "@/lib/api";
import type { DatasetStatsOut } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function DatasetPage() {
  const [stats, setStats] = useState<DatasetStatsOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasetStats()
      .then(setStats)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load dataset stats"),
      );
  }, []);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Dataset management</CardTitle>
          <CardDescription>
            HealthCareMagic-100k preprocessing status (question=`input`, answer=`output`)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && <p className="text-sm text-rose-600">{error}</p>}
          {!stats && !error && (
            <p className="text-sm text-slate-500">Loading…</p>
          )}
          {stats && (
            <>
              <div className="flex flex-wrap gap-2">
                <Badge>{stats.source_name}</Badge>
                <Badge
                  className={
                    stats.processed_available
                      ? "border-teal-200 bg-teal-50 text-teal-800"
                      : "border-amber-200 bg-amber-50 text-amber-800"
                  }
                >
                  {stats.processed_available
                    ? "Processed splits available"
                    : "Not preprocessed yet"}
                </Badge>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["Raw records", stats.total_raw],
                  ["Valid", stats.valid],
                  ["Duplicates removed", stats.duplicates_removed],
                  ["Train", stats.train_count],
                  ["Validation", stats.val_count],
                  ["Test", stats.test_count],
                ].map(([label, value]) => (
                  <div
                    key={String(label)}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-4"
                  >
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      {label}
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-slate-900">
                      {Number(value).toLocaleString()}
                    </p>
                  </div>
                ))}
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600">
                <p className="font-medium text-slate-800">CLI</p>
                <pre className="mt-2 overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
{`cd backend
source .venv/bin/activate
python -m ml preprocess`}
                </pre>
                <p className="mt-2 text-xs">
                  Splits use hash-based 80/10/10 partitioning on normalized questions
                  to avoid train/val/test leakage.
                </p>
              </div>
              {stats.stats_json && (
                <pre className="max-h-80 overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-100">
                  {JSON.stringify(stats.stats_json, null, 2)}
                </pre>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
