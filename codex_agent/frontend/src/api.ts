import type {
  Bootstrap,
  MessageResult,
  Run,
  RunEvent,
  Session,
  SessionBundle,
} from "./types";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(error?.detail || response.statusText || "请求失败");
  }
  return response.json() as Promise<T>;
}

export const api = {
  bootstrap: () => request<Bootstrap>("/api/bootstrap"),
  listSessions: () => request<Session[]>("/api/sessions"),
  createSession: (title = "新对话") =>
    request<Session>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  getSession: (sessionId: string) =>
    request<SessionBundle>(`/api/sessions/${sessionId}`),
  sendMessage: (sessionId: string, content: string) =>
    request<MessageResult>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  getRun: (runId: string) => request<Run>(`/api/runs/${runId}`),
  confirmRun: (runId: string) =>
    request<Run>(`/api/runs/${runId}/confirm`, {
      method: "POST",
      body: "{}",
    }),
  getEvents: (runId: string) =>
    request<RunEvent[]>(`/api/runs/${runId}/events`),
};
