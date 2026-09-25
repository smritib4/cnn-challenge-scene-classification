"""Render the report to a print-ready HTML page, then print it to PDF from a browser.

    python scripts/md_to_html.py reports/REPORT.md reports/REPORT.html

Deliberately dependency-free: it handles exactly the Markdown subset the report uses
(headings, bold/italic/code spans, pipe tables, ordered lists, horizontal rules) rather
than pulling in a converter. The CSS sets page margins and type size so the output
lands inside the assignment's two-page limit; adjust --pt to nudge it.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

CSS = """
@page {{ size: letter; margin: 0.7in 0.75in; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Segoe UI", Calibri, Helvetica, Arial, sans-serif;
  font-size: {pt}pt; line-height: 1.34; color: #1a1a1a; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}}
h1 {{ font-size: {h1}pt; margin: 0 0 3pt; line-height: 1.2; color: #14304d;
      font-weight: 600; letter-spacing: -0.2pt; }}
h2 {{ font-size: {h2}pt; margin: 10pt 0 4pt; color: #14304d; font-weight: 600;
      padding-bottom: 2pt; border-bottom: 1pt solid #c9d6e2;
      page-break-after: avoid; }}
p {{ margin: 0 0 6pt; }}
ol, ul {{ margin: 0 0 6pt; padding-left: 16pt; }}
li {{ margin-bottom: 4pt; }}
code {{ font-family: Consolas, "Courier New", monospace; font-size: 0.87em;
        background: #f1f4f7; padding: 0.5pt 2pt; border-radius: 2pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 9pt;
         font-size: 0.93em; page-break-inside: avoid; }}
th, td {{ border-bottom: 0.5pt solid #d8dee5; padding: 3.2pt 5pt;
          vertical-align: top; }}
th {{ border-bottom: 1pt solid #8fa5b8; text-align: left; font-weight: 600;
      color: #14304d; }}
td.r, th.r {{ text-align: right; }}
tr:first-child td {{ border-top: 0; }}
hr {{ border: none; border-top: 0.6pt solid #ccc; margin: 8pt 0 6pt; }}
.subtitle {{ margin-bottom: 11pt; color: #444; }}
"""


def inline(text: str) -> str:
    """Escape, then re-introduce the inline markup the report actually uses."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            head = split_row(lines[i])
            aligns = ["r" if c.endswith(":") else "" for c in split_row(lines[i + 1])]
            i += 2
            cells = "".join(
                f'<th class="{a}">{inline(c)}</th>' for c, a in zip(head, aligns)
            )
            rows = [f"<tr>{cells}</tr>"]
            while i < len(lines) and lines[i].strip().startswith("|"):
                body = split_row(lines[i])
                rows.append(
                    "<tr>"
                    + "".join(
                        f'<td class="{a}">{inline(c)}</td>'
                        for c, a in zip(body, aligns + [""] * len(body))
                    )
                    + "</tr>"
                )
                i += 1
            out.append("<table>" + "".join(rows) + "</table>")
            continue

        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            items: list[str] = []
            while i < len(lines) and (
                re.match(r"^\d+\.\s", lines[i].strip()) or lines[i].startswith("   ")
            ):
                s = lines[i].strip()
                if re.match(r"^\d+\.\s", s):
                    items.append(re.sub(r"^\d+\.\s+", "", s))
                elif items:
                    items[-1] += " " + s
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ol>")
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ul>")
            continue

        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(\||#|---|\d+\.\s|- )", lines[i].strip()
        ):
            para.append(lines[i].strip())
            i += 1
        cls = ' class="subtitle"' if len(out) == 1 else ""
        out.append(f"<p{cls}>{inline(' '.join(para))}</p>")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--pt", type=float, default=9.3, help="Body type size in points")
    args = parser.parse_args()

    css = CSS.format(pt=args.pt, h1=args.pt + 3.4, h2=args.pt + 1.3)
    body = render(args.source.read_text(encoding="utf-8"))
    args.dest.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CNN Challenge Report</title>"
        f"<style>{css}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {args.dest}")
    print("Open it in Chrome, press Ctrl+P, choose 'Save as PDF', margins 'Default'.")


if __name__ == "__main__":
    main()
