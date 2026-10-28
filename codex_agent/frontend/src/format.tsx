import type { ReactNode } from "react";

export function shortDate(value?: string): string {
  return value ? value.replace("T", " ").slice(5, 16) : "刚刚";
}

function inlineMarkup(value: string): ReactNode[] {
  const parts = value.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  return parts.filter(Boolean).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

export function MessageText({ value }: { value: string }) {
  const blocks: ReactNode[] = [];
  let list: string[] = [];

  const flushList = () => {
    if (!list.length) return;
    blocks.push(
      <ul key={`list-${blocks.length}`}>
        {list.map((item, index) => (
          <li key={index}>{inlineMarkup(item)}</li>
        ))}
      </ul>,
    );
    list = [];
  };

  value.split("\n").forEach((line) => {
    if (line.startsWith("- ")) {
      list.push(line.slice(2));
      return;
    }
    flushList();
    if (!line.trim()) return;
    if (line.startsWith("### ")) {
      blocks.push(<h3 key={`heading-${blocks.length}`}>{inlineMarkup(line.slice(4))}</h3>);
    } else {
      blocks.push(<p key={`paragraph-${blocks.length}`}>{inlineMarkup(line)}</p>);
    }
  });
  flushList();
  return <>{blocks}</>;
}
