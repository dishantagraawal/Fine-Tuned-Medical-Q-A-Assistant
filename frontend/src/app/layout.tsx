import type { Metadata } from "next";
import { AppShell } from "@/components/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fine-Tuned Medical Q&A Assistant",
  description:
    "Educational medical Q&A with fine-tuned LLM, RAG, and evaluation dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ fontFamily: '"IBM Plex Sans", "Segoe UI", sans-serif' }}>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
