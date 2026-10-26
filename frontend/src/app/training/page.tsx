"use client";

import { FormEvent, useEffect, useState } from "react";
import { fetchTrainingRuns, registerTrainingRun } from "@/lib/api";
import type { TrainingRunOut } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function TrainingPage() {
  const [runs, setRuns] = useState<TrainingRunOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [maxSamples, setMaxSamples] = useState(2000);
  const [epochs, setEpochs] = useState(1);
  const [hardware, setHardware] = useState("cpu");

  async function refresh() {
    try {
      setRuns(await fetchTrainingRuns());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load runs");
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function onRegister(e: FormEvent) {
    e.preventDefault();
    try {
      await registerTrainingRun({
        max_samples: maxSamples,
        epochs,
        hardware_profile: hardware,
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to register run");
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Model training</CardTitle>
          <CardDescription>
            LoRA/QLoRA fine-tuning is started from the CLI — this page registers
            configs and displays real metrics from completed runs.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form
            onSubmit={onRegister}
            className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 md:grid-cols-4"
          >
            <label className="text-sm">
              Max samples
              <Input
                type="number"
                value={maxSamples}
                onChange={(e) => setMaxSamples(Number(e.target.value))}
              />
            </label>
            <label className="text-sm">
              Epochs
              <Input
                type="number"
                step="0.1"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </label>
            <label className="text-sm">
              Hardware
              <select
                className="mt-1 flex h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"
                value={hardware}
                onChange={(e) => setHardware(e.target.value)}
              >
                <option value="cpu">cpu</option>
                <option value="mps">mps</option>
                <option value="cuda">cuda (QLoRA)</option>
              </select>
            </label>
            <div className="flex items-end">
              <Button type="submit" className="w-full">
                Register config
              </Button>
            </div>
          </form>

          <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
{`cd backend && source .venv/bin/activate
# edit configs/train.yaml (hardware_profile, max_samples, use_qlora)
python -m ml train --config configs/train.yaml`}
          </pre>

          {error && <p className="text-sm text-rose-600">{error}</p>}

          <div className="space-y-3">
            {runs.length === 0 && (
              <p className="text-sm text-slate-500">
                No training runs registered yet. Training is never started
                automatically from the UI.
              </p>
            )}
            {runs.map((run) => (
              <div
                key={run.id}
                className="rounded-xl border border-slate-200 bg-white p-4 text-sm"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Badge>{run.status}</Badge>
                  <Badge>{run.hardware_profile}</Badge>
                  <span className="text-slate-500">
                    {new Date(run.created_at).toLocaleString()}
                  </span>
                </div>
                <p className="mt-2 font-medium text-slate-800">{run.base_model}</p>
                {run.adapter_path && (
                  <p className="text-xs text-slate-500">{run.adapter_path}</p>
                )}
                {run.metrics && (
                  <pre className="mt-2 max-h-48 overflow-auto rounded bg-slate-50 p-2 text-xs">
                    {JSON.stringify(run.metrics, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
