import base64
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from html.parser import HTMLParser

from markdown_it import MarkdownIt

from .models import Chunk, Document, Edge, Relation, Section, stable_id

PARSER_VERSION = "2.1.0"


@lru_cache(maxsize=1)
def encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def token_count(text):
    return len(encoder().encode_ordinary(text))


def truncate(text, limit):
    if token_count(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def chunk_text(text, target, maximum):
    """Paragraph, then sentence, then whitespace boundaries; preserve exact text."""
    if token_count(text) <= maximum:
        return [text] if text.strip() else []
    result, start = [], 0
    boundaries = [m.end() for m in re.finditer(r"\n\s*\n|(?<=[.!?。！？])\s+|\s+", text)] + [len(text)]
    last = start
    for end in boundaries:
        if token_count(text[start:end]) > maximum:
            if last > start:
                result.append(text[start:last])
                start = last
            while token_count(text[start:end]) > maximum:
                part = truncate(text[start:end], maximum)
                if not part:
                    raise ValueError("Chunk budget cannot encode one character")
                result.append(part)
                start += len(part)
        if token_count(text[start:end]) >= target:
            result.append(text[start:end])
            start = end
        last = end
    if start < len(text):
        result.append(text[start:])
    return [part for part in result if part.strip()]


# Executable content and page chrome (menus, footers, forms) are not document text.
SKIPPED = ("script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form")


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0
        self.metadata = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name") or attrs.get("property")
            if key and attrs.get("content"):
                self.metadata[key] = attrs["content"][:2000]
        if tag in SKIPPED:
            self.skip += 1
        if self.skip:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("*")
        elif tag == "pre":
            self.parts.append("\n```\n")
        elif tag in ("p", "div", "br", "tr", "section"):
            self.parts.append("\n\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in SKIPPED and self.skip:
            self.skip -= 1
        if not self.skip and re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")
        if not self.skip and tag in ("strong", "b", "em", "i", "pre"):
            self.parts.append({"strong": "**", "b": "**", "em": "*", "i": "*", "pre": "\n```\n"}[tag])

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract(data: bytes, mime: str, max_bytes: int, with_metadata=False):
    metadata = {}
    if mime in ("application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        # Binary parsers run outside the server with a wall-clock bound, no shell or disk paths.
        result = subprocess.run([sys.executable, "-m", "graph_memory.parse_worker"],
            input=json.dumps({"data": base64.b64encode(data).decode(), "mime": mime, "limit": max_bytes}),
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
        if result.returncode:
            raise ValueError("Document parser rejected the file")
        parsed = json.loads(result.stdout)
        text, metadata = parsed["text"], parsed.get("metadata", {})
    elif mime == "message/rfc822":
        message = BytesParser(policy=policy.default).parsebytes(data)
        body = message.get_body(preferencelist=("plain", "html")) if message.is_multipart() else message
        if body is None:
            raise ValueError("Email has no readable body")
        raw = body.get_content()
        content = extract(raw.encode("utf-8"), body.get_content_type(), max_bytes)
        text = f"Subject: {message.get('Subject', '')}\nFrom: {message.get('From', '')}\nDate: {message.get('Date', '')}\n\n{content}"
    else:
        # Mislabelled legacy encodings keep their readable text instead of rejecting the whole source.
        text = data.decode("utf-8-sig", errors="replace")
        if mime == "text/html":
            parser = HTMLText()
            parser.feed(text)
            text = "".join(parser.parts)
            metadata = parser.metadata
        elif mime == "application/json":
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    if len(text.encode("utf-8")) > max_bytes * 4:
        raise ValueError("Extracted text exceeds size limit")
    if not text.strip():
        raise ValueError("No extractable text; OCR is not configured")
    return (text, metadata) if with_metadata else text


@dataclass
class Parsed:
    document: Document
    nodes: list
    edges: list
    texts: dict[str, str]


def parse_structure(text, title, document_id, settings):
    count = token_count(text)
    if count > settings.max_document_tokens:
        raise ValueError("Document exceeds token limit")
    document = Document(id=document_id, label=title, token_count=count)
    nodes, edges, texts = [document], [], {document.id: text}
    lines = text.splitlines(keepends=True)
    headings, code_lines = {}, set()
    for token in MarkdownIt().parse(text):
        if token.type in ("fence", "code_block") and token.map:
            code_lines.update(range(*token.map))
        if token.type == "heading_open" and token.map:
            line = token.map[0]
            headings[line] = (int(token.tag[1:]), re.sub(r"^\s*#+\s*|\s*#+\s*$", "", lines[line]).strip(), token.map[1])
    # Numbered headings preserve depth beyond Markdown's six heading levels.
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+(?:\.\d+)+)\.?\s+(.{1,120}?)\s*$", line)
        # Headings start with a capital and are not sentences; "2.5 times higher..." is body text.
        if match and index not in code_lines and match[2][0].isupper() and not match[2].endswith("."):
            headings[index] = (len(match[1].split(".")), match[1] + " " + match[2], index+1)
    if len(headings) > settings.max_sections:
        raise ValueError("Document exceeds section limit")
    if not headings and count <= settings.small_document_token_threshold:
        document.text, document.retrieval_leaf = text, True
        return Parsed(document, nodes, edges, texts)
    stack = [(0, document)]
    buffers = {document.id: []}
    skip_until = 0
    for index, line in enumerate(lines):
        if index < skip_until:
            continue
        if index in headings:
            level, heading, skip_until = headings[index]
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1]
            path = [node.label for _, node in stack[1:]] + [heading]
            section = Section(id=stable_id("section", f"{document_id}:{index}"), label=heading,
                              level=level, order=index, section_path=path)
            nodes.append(section)
            edges.append(Edge(source=parent.id, target=section.id, relation=Relation.CONTAINS))
            stack.append((level, section))
            buffers[section.id] = []
        else:
            buffers[stack[-1][1].id].append(line)
    for parent in list(nodes):
        body = "".join(buffers[parent.id])
        texts[parent.id] = body
        for order, part in enumerate(chunk_text(body, settings.target_chunk_tokens, settings.max_chunk_tokens)):
            path = getattr(parent, "section_path", [])
            chunk = Chunk(id=stable_id("chunk", f"{parent.id}:{order}"), label=f"{parent.label} · {order+1}",
                          text=part, token_count=token_count(part), section_path=path,
                          context_header=" > ".join([title, *path]), order=order)
            nodes.append(chunk)
            texts[chunk.id] = part
            edges.append(Edge(source=parent.id, target=chunk.id, relation=Relation.CONTAINS))
    by_id = {n.id: n for n in nodes}
    for edge in reversed(edges):
        child, parent = by_id[edge.target], by_id[edge.source]
        if child.kind == "Section":
            texts[parent.id] += "\n" + child.label + "\n" + texts[child.id]
    texts[document.id] = text
    for node in nodes:
        if node.kind == "Section":
            node.token_count = token_count(texts[node.id])
    return Parsed(document, nodes, edges, texts)
