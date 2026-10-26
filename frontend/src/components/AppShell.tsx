"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BookOpen,
  Database,
  FlaskConical,
  LayoutDashboard,
  MessageSquare,
  History,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/history", label: "History", icon: History },
  { href: "/documents", label: "Documents", icon: BookOpen },
  { href: "/dataset", label: "Dataset", icon: Database },
  { href: "/training", label: "Training", icon: FlaskConical },
  { href: "/evaluation", label: "Evaluation", icon: Activity },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#e6f4f1,_#f7fafc_45%,_#eef2ff)]">
      <div className="mx-auto flex min-h-screen max-w-7xl gap-6 px-4 py-6 md:px-6">
        <aside className="hidden w-60 shrink-0 flex-col rounded-2xl border border-slate-200/80 bg-white/90 p-4 shadow-sm backdrop-blur md:flex">
          <div className="mb-6 flex items-center gap-2 px-2">
            <LayoutDashboard className="h-5 w-5 text-teal-700" />
            <div>
              <p className="text-sm font-semibold text-slate-900">MedQA Assistant</p>
              <p className="text-xs text-slate-500">Fine-tuned · Educational</p>
            </div>
          </div>
          <nav className="flex flex-1 flex-col gap-1">
            {NAV.map((item) => {
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
                    active
                      ? "bg-teal-700 text-white"
                      : "text-slate-700 hover:bg-slate-100",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <p className="mt-4 px-2 text-[11px] leading-relaxed text-slate-500">
            Educational information only — not a substitute for professional medical advice.
          </p>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="mb-4 flex gap-2 overflow-x-auto md:hidden">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-700"
              >
                {item.label}
              </Link>
            ))}
          </div>
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </div>
  );
}
