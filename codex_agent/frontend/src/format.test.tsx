import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageText, shortDate } from "./format";

describe("shortDate", () => {
  it("formats API timestamps for the session list", () => {
    expect(shortDate("2026-08-26T11:17:47")).toBe("08-26 11:17");
  });
});

describe("MessageText", () => {
  it("renders supported message markup without injecting HTML", () => {
    render(<MessageText value={"### 计划\n- 检查 `relu`\n**确认** <script>"} />);
    expect(screen.getByRole("heading", { name: "计划" })).toBeInTheDocument();
    expect(screen.getByText("relu")).toBeInTheDocument();
    expect(screen.getByText("<script>", { exact: false })).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
  });
});
