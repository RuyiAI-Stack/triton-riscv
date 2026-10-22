import {
  Check,
  Menu,
  PanelRight,
  Plus,
  Send,
  Trash2,
  X,
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { MessageText, shortDate } from "./format";
import type {
  Bootstrap,
  Run,
  RunEvent,
  Session,
  SessionBundle,
} from "./types";

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

function EmptyConversation() {
  return (
    <div className="empty-state">
      <span className="eyebrow">Natural language workspace</span>
      <h3>从一句任务描述开始</h3>
      <p>DeepSeek Harness 会理解任务、选择 Triton-RISCV 工具，并把模型、工具调用和结果记录到会话时间线。</p>
    </div>
  );
}

export default function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<SessionBundle | null>(null);
  const [activeRun, setActiveRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<"sessions" | "inspector" | null>(null);
  const [clearedThrough, setClearedThrough] = useState(-1);
  const eventSource = useRef<EventSource | null>(null);
  const messagesEnd = useRef<HTMLDivElement | null>(null);

  const closeStream = () => {
    eventSource.current?.close();
    eventSource.current = null;
  };

  const refreshSessions = async () => {
    const rows = await api.listSessions();
    setSessions(rows);
    return rows;
  };

  const loadSession = async (sessionId: string) => {
    closeStream();
    setActiveSessionId(sessionId);
    setClearedThrough(-1);
    const nextBundle = await api.getSession(sessionId);
    setBundle(nextBundle);
    const run = nextBundle.runs[0] || null;
    setActiveRun(run);
    setEvents([]);
    setDrawer(null);
    if (run) await loadRun(run, sessionId);
  };

  const createSession = async () => {
    setError(null);
    const session = await api.createSession();
    await refreshSessions();
    await loadSession(session.id);
  };

  const openStream = (runId: string, sessionId: string, after: number) => {
    closeStream();
    const stream = new EventSource(`/api/runs/${runId}/events/stream?after=${after}`);
    eventSource.current = stream;
    stream.onmessage = async (message) => {
      const item = JSON.parse(message.data) as RunEvent;
      setEvents((current) => [...current, item]);
      try {
        const run = await api.getRun(runId);
        setActiveRun(run);
        if (terminalStatuses.has(run.status)) {
          closeStream();
          setBundle(await api.getSession(sessionId));
        }
      } catch (streamError) {
        setError(streamError instanceof Error ? streamError.message : "读取运行状态失败");
        closeStream();
      }
    };
    stream.onerror = () => {
      setError("实时日志连接已中断，可以重新打开该会话继续查看结果。");
      closeStream();
    };
  };

  const loadRun = async (run: Run, sessionId: string) => {
    const [currentRun, runEvents] = await Promise.all([api.getRun(run.id), api.getEvents(run.id)]);
    setActiveRun(currentRun);
    setEvents(runEvents);
    if (["queued", "running"].includes(currentRun.status)) {
      const after = runEvents.at(-1)?.sequence ?? -1;
      openStream(run.id, sessionId, after);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const start = async () => {
      try {
        const [initialBootstrap, initialSessions] = await Promise.all([
          api.bootstrap(),
          api.listSessions(),
        ]);
        if (cancelled) return;
        setBootstrap(initialBootstrap);
        setSessions(initialSessions);
        if (initialSessions.length) await loadSession(initialSessions[0].id);
        else await createSession();
      } catch (startError) {
        if (!cancelled) setError(startError instanceof Error ? startError.message : "平台启动失败");
      }
    };
    void start();
    return () => {
      cancelled = true;
      closeStream();
    };
  }, []);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [bundle?.messages.length]);

  const timeline = events.filter((event) => event.event_type !== "output");
  const logLines = events
    .filter((event) => ["output", "harness-event"].includes(event.event_type) && event.sequence > clearedThrough)
    .map((event) => event.payload.line || event.payload.detail || event.payload.message || "");

  const sendMessage = async (content: string) => {
    if (!content.trim() || !activeSessionId || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.sendMessage(activeSessionId, content.trim());
      setBundle((current) => current ? {
        ...current,
        messages: [...current.messages, result.user_message, result.assistant_message],
        runs: result.run ? [result.run, ...current.runs] : current.runs,
      } : current);
      if (result.run) {
        setActiveRun(result.run);
        setEvents(await api.getEvents(result.run.id));
      }
      await refreshSessions();
    } catch (sendError) {
      setError(sendError instanceof Error ? sendError.message : "发送失败");
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const content = input;
    setInput("");
    void sendMessage(content);
  };

  const inputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const confirmRun = async (runId: string) => {
    if (!activeSessionId) return;
    setBusy(true);
    setError(null);
    try {
      const run = await api.confirmRun(runId);
      setActiveRun(run);
      setBundle((current) => current ? {
        ...current,
        runs: current.runs.map((item) => item.id === runId ? run : item),
      } : current);
      openStream(runId, activeSessionId, events.at(-1)?.sequence ?? -1);
    } catch (confirmError) {
      setError(confirmError instanceof Error ? confirmError.message : "执行确认失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`app-shell ${drawer ? `show-${drawer}` : ""}`}>
      <header className="topbar">
        <button className="mobile-tool" type="button" title="打开任务会话" aria-label="打开任务会话" onClick={() => setDrawer("sessions")}>
          <Menu size={18} />
        </button>
        <div className="brand-block">
          <div className="brand-mark">RV</div>
          <div>
            <h1>Triton-RISCV Agent</h1>
            <p>算子开发与编译验证工作台</p>
          </div>
        </div>
        <div className="system-state"><span className="status-dot" /><span>{bootstrap?.harness.api_configured ? "Harness 已配置" : "等待模型配置"}</span></div>
        <button className="mobile-tool" type="button" title="打开任务详情" aria-label="打开任务详情" onClick={() => setDrawer("inspector")}>
          <PanelRight size={18} />
        </button>
      </header>

      <aside className="sessions-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Workspace</span><h2>任务会话</h2></div>
          <button className="icon-button" type="button" title="新建任务" aria-label="新建任务" onClick={() => void createSession()}>
            <Plus size={18} />
          </button>
        </div>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              className={`session-button ${session.id === activeSessionId ? "active" : ""}`}
              key={session.id}
              type="button"
              onClick={() => void loadSession(session.id)}
            >
              <strong>{session.title}</strong><span>{shortDate(session.updated_at)}</span>
            </button>
          ))}
        </div>
        <div className="inventory-strip">
          <div><strong>{bootstrap?.operators.operators || 0}</strong><span>算子</span></div>
          <div><strong>{bootstrap?.project.total_targets || 0}</strong><span>验证目标</span></div>
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-heading">
          <div><span className="eyebrow">Conversation</span><h2>{bundle?.session.title || "新对话"}</h2></div>
          <span className="mode-badge">DeepSeek Harness</span>
        </div>
        {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" title="关闭错误提示" aria-label="关闭错误提示" onClick={() => setError(null)}><X size={16} /></button></div>}
        <div className="messages" aria-live="polite">
          {!bundle?.messages.length ? <EmptyConversation /> : bundle.messages.map((message) => {
            const run = message.metadata.run_id ? bundle.runs.find((item) => item.id === message.metadata.run_id) : null;
            return (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-role">{message.role === "user" ? "You" : "Agent"}</div>
                <div className="message-body">
                  <MessageText value={message.content} />
                  {run?.status === "awaiting-confirmation" && (
                    <div className="confirmation">
                      <button className="primary-button" type="button" disabled={busy} onClick={() => void confirmRun(run.id)}>
                        <Check size={15} /> 确认执行
                      </button>
                      <span>确认后才会调用模型和工具</span>
                    </div>
                  )}
                </div>
              </article>
            );
          })}
          <div ref={messagesEnd} />
        </div>
        <div className="suggestions">
          {bootstrap?.suggestions.map((suggestion) => (
            <button className="suggestion-button" type="button" title={suggestion} key={suggestion} onClick={() => setInput(suggestion)}>{suggestion}</button>
          ))}
        </div>
        <form className="composer" onSubmit={submit}>
          <textarea value={input} rows={2} maxLength={12000} placeholder="描述你想检查、开发或分析的 Triton-RISCV 任务..." aria-label="任务描述" onChange={(event) => setInput(event.target.value)} onKeyDown={inputKeyDown} />
          <button className="send-button" type="submit" title="发送任务" aria-label="发送任务" disabled={busy || !input.trim()}><Send size={18} /></button>
        </form>
      </main>

      <aside className="inspector-panel">
        <div className="panel-heading"><div><span className="eyebrow">Agent State</span><h2>任务详情</h2></div></div>
        <section className="inspector-section">
          <h3>Agent 运行时</h3>
          <dl className="fact-list">
            <div><dt>框架</dt><dd>{bootstrap?.harness.runtime || "加载中"}</dd></div>
            <div><dt>模型</dt><dd>{bootstrap?.harness.model || "未配置"}</dd></div>
            <div><dt>API</dt><dd>{bootstrap?.harness.api_configured ? "已配置" : "未配置"}</dd></div>
          </dl>
        </section>
        <section className="inspector-section">
          <div className="section-title-row"><h3>运行时间线</h3><span className={`run-state ${activeRun?.status || ""}`}>{activeRun?.status || "空闲"}</span></div>
          <ol className="timeline">
            {!timeline.length ? <li className="muted">确认任务后显示实时阶段</li> : timeline.map((event, index) => {
              const label = event.payload.message || event.payload.command || event.event_type;
              const active = index === timeline.length - 1 && activeRun?.status === "running";
              return <li className={active ? "active" : "done"} key={event.id}><strong>{event.event_type}</strong><br />{String(label)}</li>;
            })}
          </ol>
        </section>
        <section className="inspector-section log-section">
          <div className="section-title-row">
            <h3>实时输出</h3>
            <button className="icon-text-button" type="button" title="清空当前日志视图" onClick={() => setClearedThrough(events.at(-1)?.sequence ?? -1)}><Trash2 size={14} />清空</button>
          </div>
          <pre className="run-log">{logLines.length ? logLines.join("\n") : "尚未产生运行输出。"}</pre>
        </section>
      </aside>

      {drawer && <button className="scrim" type="button" aria-label="关闭侧栏" onClick={() => setDrawer(null)} />}
    </div>
  );
}
