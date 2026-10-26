"""Run each historical frontend case through the neutral contract boundary.

Unsupported control flow and layouts must identify their contract restriction;
accepted cases must survive the independent verifier as buffer-form standard IR.
The source functions remain separate cases so one rejected function cannot hide
an unrelated case later in the same input file.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

FUNCTION = re.compile(
    r"\b((?:func|tt)\.func)\s+(?:(?:public|private)\s+)?@(\w+)\("
)


def functions(text):
    # CHECK/RUN comments can themselves contain function spellings and braces.
    text = re.sub(r"//[^\n]*", "", text)
    matches = list(FUNCTION.finditer(text))
    found = {}
    for index, match in enumerate(matches):
        limit = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        cursor, parens = match.end() - 1, 0
        while cursor < limit:
            char = text[cursor]
            parens += (char == "(") - (char == ")")
            if char == "{" and parens == 0:
                depth, closing = 1, cursor + 1
                while depth and closing < limit:
                    depth += (text[closing] == "{") - (text[closing] == "}")
                    closing += 1
                if re.search(r"\battributes\s*$", text[match.start() : cursor]):
                    cursor = closing
                    continue
                found[match[2]] = (
                    text[match.start() : cursor].strip(),
                    text[match.start() : closing],
                )
                break
            cursor += 1
        else:
            found[match[2]] = (text[match.start() : limit].strip(), None)
    return found


def isolated_module(source, name):
    available = functions(source)
    _, body = available[name]
    if body is None:
        return None
    declarations = []
    for callee in set(re.findall(r"(?:func|tt)\.call\s+@(\w+)", body)):
        header, _ = available[callee]
        header = re.sub(
            r"^func\.func\s+(?:(?:public|private)\s+)?",
            "func.func private ",
            header,
        )
        declarations.append(header)
    aliases = "\n".join(
        line for line in source.splitlines() if line.startswith("#")
    )
    globals_ = "\n".join(
        line
        for line in source.splitlines()
        if line.strip().startswith("memref.global ")
    )
    return (
        aliases
        + "\nmodule {\n"
        + globals_
        + "\n"
        + "\n".join(declarations)
        + "\n"
        + body
        + "\n}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("opt")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    source = args.source.read_text()
    expectations = dict(
        re.findall(r"^// CONTRACT-CASE: (\w+) = (.*)$", source, re.M)
    )
    cases = {
        name
        for name, (_, body) in functions(source).items()
        if body is not None
    }
    if not args.record:
        assert cases == expectations.keys(), (
            args.source,
            cases ^ expectations.keys(),
        )
    for name in sorted(cases):
        path = Path(str(args.prefix) + "." + name + ".mlir")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(isolated_module(source, name))
        result = subprocess.run(
            [
                args.opt,
                str(path),
                "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias",
                "--kernel-contract-verify",
            ],
            text=True,
            capture_output=True,
        )
        if args.record:
            print(
                json.dumps(
                    {
                        "file": args.source.name,
                        "name": name,
                        "status": result.returncode,
                        "error": result.stderr.splitlines()[0]
                        if result.stderr
                        else "",
                        "incomplete": any(
                            re.search(r"(?:^|, )complete = false", graph)
                            for graph in re.findall(
                                r"contract\.compute_graph = \{(.*?)\}",
                                result.stdout,
                            )
                        ),
                    }
                )
            )
            continue
        expected = expectations[name]
        if expected.startswith("reject: "):
            assert result.returncode > 0, (
                args.source,
                name,
                "expected contract rejection",
                result.stderr,
            )
            assert expected.removeprefix("reject: ") in result.stderr, (
                args.source,
                name,
                result.stderr,
            )
            assert (
                "LLVM ERROR" not in result.stderr
                and "Stack dump" not in result.stderr
            ), result.stderr
            continue
        assert result.returncode == 0, (args.source, name, result.stderr)
        output = result.stdout
        assert (
            "contract.version = 1" in output
            and "contract.compute_graph" in output
        ), name
        assert "bosc_ame." not in output and "plan." not in output, name
        assert "tensor<" not in output and "bufferization." not in output, name
        graphs = re.findall(r"contract\.compute_graph = \{(.*?)\}", output)
        assert len(graphs) == 1, (name, output)
        incomplete = re.search(r"(?:^|, )complete = false", graphs[0])
        if expected == "incomplete":
            assert incomplete and "limitation =" in graphs[0], name
        else:
            assert expected == "complete" and not incomplete, (name, output)
        reported = subprocess.run(
            [
                args.opt,
                str(path),
                "--triton-to-linalg-experimental=contract=true "
                "caller-contract=args_aligned64_nonalias dump-facts=true",
            ],
            text=True,
            capture_output=True,
        )
        assert reported.returncode == 0, (name, reported.stderr)
        assert reported.stdout == output, (
            name,
            "dump-facts mutated the contract",
        )
        assert reported.stderr.strip() == output.strip(), (
            name,
            "dump-facts is not the contract IR",
        )
        Path(str(args.prefix) + "." + name + ".contract.mlir").write_text(
            output
        )


if __name__ == "__main__":
    main()
