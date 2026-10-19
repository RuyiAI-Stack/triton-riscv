export type RunStatus =
  | "awaiting-confirmation"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageMetadata {
  run_id?: string;
  task?: {
    intent: string;
    runtime?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  metadata: MessageMetadata;
  created_at: string;
}

export interface Run {
  id: string;
  session_id: string;
  status: RunStatus;
  command?: string;
  created_at: string;
  updated_at: string;
}

export interface RunEvent {
  id: string;
  run_id: string;
  sequence: number;
  event_type: string;
  payload: {
    line?: string;
    message?: string;
    command?: string;
    detail?: string;
    [key: string]: unknown;
  };
  created_at: string;
}

export interface SessionBundle {
  session: Session;
  messages: Message[];
  runs: Run[];
}

export interface Bootstrap {
  operators: { operators?: number; [key: string]: unknown };
  project: { total_targets?: number; [key: string]: unknown };
  harness: {
    runtime: string;
    provider: string;
    model: string;
    api_configured: boolean;
    remote_configured: boolean;
  };
  suggestions: string[];
}

export interface MessageResult {
  user_message: Message;
  assistant_message: Message;
  run: Run | null;
}
