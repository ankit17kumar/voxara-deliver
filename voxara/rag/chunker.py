"""Split policy markdown into small, self-contained chunks.

Voice answers must be short, so we chunk by `##` section rather than by a
fixed token window: each section is one rule the agent can cite on its own.
The section title is prepended to the chunk text so retrieval sees it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def chunk_markdown(raw: str, fallback_id: str, max_chars: int = 700) -> list[Chunk]:
    meta, body = _parse_front_matter(raw)
    doc_id = meta.get("id", fallback_id)
    doc_title = ""
    chunks: list[Chunk] = []
    section, buf = None, []

    def flush():
        text = " ".join(" ".join(buf).split())
        if section and text:
            # Very long sections are split on sentence boundaries.
            parts, cur = [], ""
            for sent in re.split(r"(?<=[.?!])\s+", text):
                if len(cur) + len(sent) > max_chars and cur:
                    parts.append(cur.strip())
                    cur = ""
                cur += sent + " "
            if cur.strip():
                parts.append(cur.strip())
            for i, p in enumerate(parts):
                cid = f"{doc_id}#{_slug(section)}" + (f"-{i}" if i else "")
                chunks.append(Chunk(cid, doc_id, f"{doc_title} / {section}",
                                    f"{section}. {p}", {**meta, "body": p}))

    for line in body.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


def load_kb(kb_dir: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        out.extend(chunk_markdown(p.read_text(encoding="utf-8"), p.stem))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
