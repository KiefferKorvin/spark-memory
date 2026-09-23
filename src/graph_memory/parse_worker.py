"""Disposable binary-document parser. Never accepts filesystem paths or fetches links."""
import base64
import io
import json
import sys
import zipfile


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    except ImportError:
        pass  # Windows has a parent-enforced process timeout; deploy Linux for memory isolation.
    item = json.loads(sys.stdin.read())
    data, limit = base64.b64decode(item["data"], validate=True), item["limit"]
    parts, size = [], 0
    metadata = {}

    def append(text):
        nonlocal size
        size += len(text.encode("utf-8"))
        if size > limit * 4:
            raise ValueError("Extracted text too large")
        parts.append(text)

    if item["mime"] == "application/pdf":
        from pypdf import PdfReader
        if not data.startswith(b"%PDF-"):
            raise ValueError("Not PDF")
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted or len(reader.pages) > 3000:
            raise ValueError("Encrypted or excessive PDF")
        metadata = {str(k): str(v)[:2000] for k, v in (reader.metadata or {}).items()}
        # Plain mode keeps sentences intact (layout mode pads columns with spaces and inflates tokens).
        # Page markers are HTML comments, not headings, so pages do not become sections; blank
        # pages (separators, figures) are skipped and extract() still rejects a PDF with no text at all.
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                append(f"<!-- page {number} -->\n{text}")
    else:
        from docx import Document
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > limit*10 or len(archive.infolist()) > 2000:
                raise ValueError("Excessive archive expansion")
            if any("vbaProject" in name for name in archive.namelist()):
                raise ValueError("Macros are not supported")
        document = Document(io.BytesIO(data))
        props = document.core_properties
        metadata = {k: str(getattr(props, k)) for k in ("title", "author", "subject", "created", "modified", "keywords", "language") if getattr(props, k)}
        for block in document.iter_inner_content():
            if hasattr(block, "text"):
                style = block.style.name if block.style else ""
                prefix = "#" * int(style.split()[-1]) + " " if style.startswith("Heading ") and style.split()[-1].isdigit() else ""
                if not prefix and "List" in style:
                    prefix = "1. " if "Number" in style else "- "
                runs = []
                for run in block.runs:
                    value = run.text
                    if value.strip() and run.bold:
                        value = "**" + value + "**"
                    if value.strip() and run.italic:
                        value = "*" + value + "*"
                    runs.append(value)
                append(prefix + ("".join(runs) or block.text))
            else:
                rows = ["| " + " | ".join(cell.text.replace("|", "\\|").replace("\n", "<br>") for cell in row.cells) + " |" for row in block.rows]
                if rows:
                    rows.insert(1, "| " + " | ".join("---" for _ in block.rows[0].cells) + " |")
                append("\n".join(rows))
    print(json.dumps({"text": "\n\n".join(parts), "metadata": metadata}, ensure_ascii=True))


if __name__ == "__main__":
    main()
