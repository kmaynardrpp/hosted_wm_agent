import React, { useCallback, useRef, useState } from "react";
const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  const fileRef = useRef(null);

  const removeFile = (i) => setFiles((prev) => prev.filter((_, idx) => idx !== i));
  const pickFiles = () => fileRef.current?.click();

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files || []);
    if (dropped.length) setFiles((p) => [...p, ...dropped]);
  }, []);

  const onPaste = useCallback((e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const pasted = [];
    for (const it of items) if (it.kind === "file") {
      const f = it.getAsFile(); if (f) pasted.push(f);
    }
    if (pasted.length) setFiles((p) => [...p, ...pasted]);
  }, []);

  const send = useCallback(async () => {
    const trimmed = prompt.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    try {
      const fd = new FormData();
      fd.append("prompt", trimmed);
      for (const f of files) fd.append("files", f, f.name);

      const res = await fetch(`${API_BASE}/api/run`, { method: "POST", body: fd });
      if (!res.ok) {
        const t = await res.text();
        setMessages((prev) => [...prev, { role: "assistant", text: `Error: ${res.status}. ${t}` }]);
      } else {
        const data = await res.json();
        setMessages((prev) => [...prev, {
          role: "assistant",
          text: data.summary || "Report ready.",
          artifacts: Array.isArray(data.artifacts) ? data.artifacts : [],
          logs: data.logs || "",
        }]);
      }
    } catch (e) {
      setMessages((prev) => [...prev, { role: "assistant", text: `Network error: ${e.message || e}` }]);
    } finally {
      setBusy(false);
      setPrompt("");
      setFiles([]);
      inputRef.current?.focus();
    }
  }, [prompt, files, busy]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col" onPaste={onPaste}>
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="msg-wrap mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-blue-600" />
            <h1 className="font-semibold">InfoZone Web Chat</h1>
          </div>
          <a className="text-sm text-blue-600 hover:underline" href={`${API_BASE}/docs`} target="_blank">API docs</a>
        </div>
      </header>

      <main className="flex-1 msg-wrap mx-auto w-full px-4 py-6"
            onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
        <div className="space-y-6">
          <div className="space-y-4">
            {messages.map((m, i) => <MessageBubble key={i} {...m} apiBase={API_BASE} />)}
          </div>

          <div className="border rounded-2xl shadow-soft bg-white">
            <div className="p-3 flex flex-wrap gap-2">
              {files.map((f, i) => (
                <span key={i} className="inline-flex items-center gap-2 text-xs px-2 py-1 bg-gray-100 rounded-full">
                  <svg width="14" height="14" viewBox="0 0 24 24"><path fill="currentColor" d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/></svg>
                  {f.name}
                  <button onClick={() => removeFile(i)} className="hover:text-red-600">×</button>
                </span>
              ))}
            </div>

            <div className="px-3 pb-3">
              <textarea
                ref={inputRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={busy ? "Running…" : "Type a prompt… (Shift+Enter for newline)"}
                className="w-full h-28 resize-none outline-none p-3 rounded-xl bg-gray-50 focus:bg-white"
                disabled={busy}
              />
              <div className="mt-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <input ref={fileRef} type="file" multiple className="hidden"
                         onChange={(e) => setFiles((p) => [...p, ...Array.from(e.target.files || [])])} />
                  <button onClick={pickFiles} className="text-sm px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200">
                    Attach files
                  </button>
                  <span className="text-xs text-gray-500">Drop or paste files, too</span>
                </div>
                <button
                  onClick={send}
                  disabled={busy || !prompt.trim()}
                  className="px-4 py-2 rounded-xl bg-blue-600 text-white disabled:opacity-50"
                >{busy ? "Running…" : "Send"}</button>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="text-xs text-center text-gray-500 py-6">Local-only • Displays PDFs & PNGs from your generator</footer>
    </div>
  );
}

function MessageBubble({ role, text, artifacts = [], logs = "", apiBase }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] w-fit bubble px-4 py-3 shadow-soft ${isUser ? "bg-blue-600 text-white" : "bg-white border"}`}>
        <div className="whitespace-pre-wrap text-sm">{text}</div>
        {!isUser && artifacts?.length > 0 && (
          <div className="mt-3 space-y-3">
            {artifacts.map((a, i) => <Artifact key={i} a={a} apiBase={apiBase} />)}
          </div>
        )}
        {!isUser && logs && (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs text-gray-500">View logs</summary>
            <pre className="text-[11px] p-2 bg-gray-50 rounded-md overflow-x-auto whitespace-pre-wrap">{logs}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

function Artifact({ a, apiBase }) {
  const url = absolutify(a?.url, apiBase);
  const filename = a?.filename || url;
  const ext = (url.split(".").pop() || "").toLowerCase();

  if (ext === "pdf") {
    return (
      <div className="border rounded-lg overflow-hidden">
        <div className="text-xs px-3 py-2 bg-gray-100 flex items-center justify-between">
          <span>{filename}</span>
          <a href={url} target="_blank" className="text-blue-600 hover:underline">Open</a>
        </div>
        <iframe src={`${url}#toolbar=0`} className="art-iframe" />
      </div>
    );
  }
  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="text-xs px-3 py-2 bg-gray-100 flex items-center justify-between">
        <span>{filename}</span>
        <a href={url} target="_blank" className="text-blue-600 hover:underline">Open</a>
      </div>
      {/* eslint-disable-next-line jsx-a11y/alt-text */}
      <img src={url} alt={filename} className="max-h-[480px] w-auto" />
    </div>
  );
}

function absolutify(u, base) {
  if (!u) return "";
  if (u.startsWith("http://") || u.startsWith("https://")) return u;
  if (u.startsWith("/")) return `${base}${u}`;
  return u;
}
