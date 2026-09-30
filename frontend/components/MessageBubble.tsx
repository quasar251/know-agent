"use client";

import { FileText, User, Bot } from "lucide-react";
import ThinkingChain, { type ToolEvent } from "@/components/ThinkingChain";
import ReportView from "@/components/ReportView";
import ExportActions from "@/components/ExportActions";
import type { Message, SourceRef } from "@/lib/conversationStore";

export default function MessageBubble({
  message,
  prevUserMessage,
}: {
  message: Message;
  /** v3.1: pass-through to ExportActions → ShareCardDialog renders "Q: ..." sub-heading. */
  prevUserMessage?: string;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[85%] items-start gap-2">
          <div className="rounded-2xl rounded-tr-sm bg-accent px-4 py-2.5 text-white whitespace-pre-wrap break-words">
            {message.content}
          </div>
          <div className="mt-1 flex h-7 w-7 flex-none items-center justify-center rounded-full bg-accent/15 text-accent">
            <User className="h-4 w-4" />
          </div>
        </div>
      </div>
    );
  }

  // assistant
  const hasContent = message.content && message.content.length > 0;
  const hasTools = message.tools && message.tools.length > 0;
  const sources = message.sources ?? [];
  const hasSources = sources.length > 0;
  const streaming = !!message.streaming;

  const showInitialThinking = streaming && !hasTools && !hasContent && !message.error;
  const showWritingHint =
    streaming &&
    hasTools &&
    !hasContent &&
    !message.error &&
    message.tools.every((t) => t.status !== "running");

  return (
    <div className="flex justify-start">
      <div className="flex max-w-full items-start gap-2 w-full">
        <div className="mt-1 flex h-7 w-7 flex-none items-center justify-center rounded-full bg-fg/10 text-fg/80">
          <Bot className="h-4 w-4" />
        </div>
        <div className="flex-1 space-y-3 min-w-0">
          {showInitialThinking && <ThinkingPlaceholder label="正在思考" />}

          {hasTools && <ThinkingChain events={message.tools} />}

          {showWritingHint && <ThinkingPlaceholder label="正在撰写报告" />}

          {message.error && (
            <div className="rounded-md border border-red-300/40 bg-red-50/60 p-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">
              ⚠️ {message.error}
            </div>
          )}

          {hasContent && (
            <ReportView markdown={message.content} streaming={streaming} />
          )}

          {!hasContent && !streaming && !message.error && (
            <div className="text-sm text-muted">（无内容）</div>
          )}

          {hasSources && <SourcesList sources={sources} content={message.content} />}

          {hasContent && !streaming && (
            <ExportActions
              markdown={message.content}
              cost={message.cost_usd ?? null}
              question={prevUserMessage}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function SourcesList({ sources, content }: { sources: SourceRef[]; content: string }) {
  // A source counts as "cited" when the answer text mentions its filename —
  // a cheap, robust signal that the model explicitly referenced it.
  const cited = (fn: string) => !!content && content.includes(fn);
  const anyCited = sources.some((s) => cited(s.filename));

  return (
    <div className="rounded-xl border border-fg/10 bg-fg/[0.02] p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted">
        <FileText className="h-3.5 w-3.5" />
        <span>参考来源（{sources.length}）</span>
      </div>
      <ul className="space-y-1.5">
        {sources.map((s) => {
          const isCited = cited(s.filename);
          return (
            <li key={s.filename} className="flex items-center gap-2 text-xs">
              <span className={isCited ? "text-accent" : "text-muted"}>
                {isCited ? "★" : "·"}
              </span>
              <span
                className={`min-w-0 truncate ${
                  isCited ? "font-medium text-accent" : "text-fg/70"
                }`}
                title={s.filename}
              >
                {s.filename}
              </span>
              <span className="ml-auto flex-none text-muted">
                相关度 {s.score.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
      {anyCited && (
        <div className="mt-2 text-[11px] text-muted">★ 表示正文中明确引用的来源</div>
      )}
    </div>
  );
}

function ThinkingPlaceholder({ label }: { label: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-xl border border-fg/10 bg-fg/[0.02] px-3 py-2 text-sm text-muted">
      <span className="relative inline-flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
      </span>
      <span>{label}</span>
      <span className="inline-flex gap-0.5">
        <span className="h-1 w-1 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
        <span className="h-1 w-1 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
        <span className="h-1 w-1 animate-bounce rounded-full bg-muted" />
      </span>
    </div>
  );
}
