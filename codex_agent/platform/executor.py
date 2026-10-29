"""Confirmation-gated DeepSeek Harness execution for FastAPI runs."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codex_agent.harness import HarnessAgent
from codex_agent.platform.store import PlatformStore


class HarnessRunExecutor:
    def __init__(
        self,
        repo_root: Path,
        store: PlatformStore,
        agent: HarnessAgent,
        workers: int = 2,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.store = store
        self.agent = agent
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dsh-agent")
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def confirm(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if run["status"] == "awaiting-confirmation":
            self._validate_request(run["request"])
            self.store.update_run(run_id, status="queued", phase="confirmed")
            self.store.add_event(run_id, "confirmed", {"message": "用户已确认调用 Harness"})
            with self._lock:
                if run_id not in self._active:
                    self._active.add(run_id)
                    self.pool.submit(self._execute, run_id)
        return self.store.get_run(run_id)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.agent.close()

    @staticmethod
    def _validate_request(request: dict) -> None:
        if request.get("action") != "deepseek-harness":
            raise ValueError("run request is not a DeepSeek Harness task")
        task = request.get("task")
        if not isinstance(task, str) or not task.strip() or "\x00" in task:
            raise ValueError("invalid Harness task")
        session_id = request.get("harness_session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("invalid Harness session id")

    @staticmethod
    def _notification_payload(notification: dict[str, Any]) -> dict[str, Any]:
        method = str(notification.get("method", "harness.notification"))
        payload = notification.get("payload", {})
        preview = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(preview) > 4000:
            preview = preview[:4000] + "..."
        return {"message": method, "detail": preview}

    def _execute(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        request = run["request"]
        self.store.update_run(run_id, status="running", phase="harness-running")
        self.store.add_event(
            run_id,
            "started",
            {
                "message": "DeepSeek Harness agent loop started",
                "model": self.agent.settings.model,
            },
        )
        event_count = 0

        def record_event(notification: dict[str, Any]) -> None:
            nonlocal event_count
            event_count += 1
            self.store.add_event(
                run_id,
                "harness-event",
                self._notification_payload(notification),
            )

        try:
            result = self.agent.run(
                request["task"],
                session_id=request["harness_session_id"],
                on_event=record_event,
            )
            for line in result.final_response.splitlines()[-80:]:
                self.store.add_event(run_id, "output", {"line": line})
            final = {
                "harness_session_id": result.session_id,
                "finish_reason": result.finish_reason,
                "event_count": event_count,
            }
            self.store.update_run(run_id, status="completed", phase="completed", result=final)
            self.store.add_event(run_id, "completed", final)
            self.store.add_message(
                run["session_id"],
                "assistant",
                result.final_response,
                {
                    "view": "harness-result",
                    "run_id": run_id,
                    "harness_session_id": result.session_id,
                },
            )
        except Exception as error:
            final = {"error": str(error), "event_count": event_count}
            self.store.update_run(run_id, status="failed", phase="failed", result=final)
            self.store.add_event(run_id, "failed", {"message": str(error)})
            self.store.add_message(
                run["session_id"],
                "assistant",
                f"DeepSeek Harness 执行失败：`{error}`",
                {"view": "harness-error", "run_id": run_id},
            )
        finally:
            with self._lock:
                self._active.discard(run_id)
