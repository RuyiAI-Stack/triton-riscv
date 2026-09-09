import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

function jsonResponse(value: unknown, ok = true): Response {
  return {
    ok,
    statusText: ok ? "OK" : "Bad Request",
    json: async () => value,
  } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("FastAPI client", () => {
  it("sends natural-language tasks to the active session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run: null }));
    vi.stubGlobal("fetch", fetchMock);

    await api.sendMessage("session-1", "验证 relu_and_mul");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/session-1/messages",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ content: "验证 relu_and_mul" }),
      }),
    );
  });

  it("uses a separate confirmation request before execution", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "run-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await api.confirmRun("run-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs/run-1/confirm",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("surfaces FastAPI validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "message cannot be empty" }, false)),
    );

    await expect(api.sendMessage("session-1", "")).rejects.toThrow(
      "message cannot be empty",
    );
  });
});
