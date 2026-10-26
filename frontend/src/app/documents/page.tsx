"use client";

import { useEffect, useState } from "react";
import {
  deleteDocument,
  fetchDocuments,
  uploadDocument,
} from "@/lib/api";
import type { DocumentOut } from "@/lib/types";
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

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function refresh() {
    try {
      setDocs(await fetchDocuments());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function onUpload(file: File | null) {
    if (!file) return;
    setUploading(true);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function onDelete(id: string) {
    try {
      await deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Medical document library</CardTitle>
        <CardDescription>
          Upload PDFs for optional RAG retrieval with page citations.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Input
            type="file"
            accept="application/pdf"
            disabled={uploading}
            onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
          />
          {uploading && <span className="text-sm text-slate-500">Indexing…</span>}
        </div>
        {error && <p className="text-sm text-rose-600">{error}</p>}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b text-slate-500">
              <tr>
                <th className="py-2 pr-3">Name</th>
                <th className="py-2 pr-3">Pages</th>
                <th className="py-2 pr-3">Chunks</th>
                <th className="py-2 pr-3">Status</th>
                <th className="py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id} className="border-b border-slate-100">
                  <td className="py-3 pr-3 font-medium text-slate-800">
                    {d.original_name}
                  </td>
                  <td className="py-3 pr-3">{d.page_count}</td>
                  <td className="py-3 pr-3">{d.chunk_count}</td>
                  <td className="py-3 pr-3">
                    <Badge>{d.status}</Badge>
                  </td>
                  <td className="py-3">
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={() => onDelete(d.id)}
                    >
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
              {docs.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-6 text-slate-500">
                    No documents uploaded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
