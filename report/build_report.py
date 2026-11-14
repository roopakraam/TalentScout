"""Build the Word version of the report from the Markdown chapters.

    python report/build_report.py            # -> report/TalentScout_v2_Report.docx

Pandoc and LaTeX are not available on the project machine, so this small builder
turns a restricted Markdown dialect into a .docx with python-docx:

* ``#``, ``##``, ``###`` headings (every ``#`` starts a new page and is numbered 1, 2, ...)
* paragraphs with **bold**, *italic* and `code`
* bullet (``-``) and numbered (``1.``) lists
* pipe tables, preceded by ``Table: caption {#tab:label}``
* figures: ``![caption](path/to/figure.png){#fig:label}``
* display equations on one line: ``$$ latex $$ {#eq:label}``, rendered with matplotlib mathtext
* cross-references ``@fig:label``, ``@tab:label``, ``@eq:label`` and ``@sec:label``
  (a heading gets a label with ``{#sec:label}``)

Chapters are the files ``report/chapters/*.md`` in name order; ``report/front.json``
holds the title page. ``--paper`` builds the short paper from ``paper/chapters`` instead
(compact layout: title block on page 1, no contents page, no page break per section);
``report/front.json``
holds the title page.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHAPTERS = HERE / "chapters"
OUT = HERE / "TalentScout_v2_Report.docx"
COMPACT = False                 # set by --paper

BODY_FONT = "Times New Roman"
FIGURE_SCALE = 0.88          # figure widths in the chapters are multiplied by this
INK = RGBColor(0x1A, 0x1A, 0x1A)
LABEL_RE = re.compile(r"\{#((?:fig|tab|eq|sec):[\w-]+)[^}]*\}")
REF_RE = re.compile(r"@((?:fig|tab|eq|sec):[\w-]+)")


# --------------------------------------------------------------------------- numbering pass
def is_unnumbered(title: str) -> bool:
    return title.lower().startswith(("abstract", "references", "appendix"))


def collect_labels(blocks_by_file: list[list[str]]) -> dict[str, str]:
    """First pass: give every label its number, in reading order."""
    numbers: dict[str, str] = {}
    counters = {"fig": 0, "tab": 0, "eq": 0}
    chapter = section = subsection = 0
    for lines in blocks_by_file:
        for line in lines:
            m = LABEL_RE.search(line)
            if line.startswith("# "):
                if is_unnumbered(LABEL_RE.sub("", line[2:]).strip()):
                    continue
                chapter += 1
                section = subsection = 0
                if m:
                    numbers[m.group(1)] = f"{chapter}"
            elif line.startswith("## "):
                section += 1
                subsection = 0
                if m:
                    numbers[m.group(1)] = f"{chapter}.{section}"
            elif line.startswith("### "):
                subsection += 1
                if m:
                    numbers[m.group(1)] = f"{chapter}.{section}.{subsection}"
            elif m:
                kind = m.group(1).split(":")[0]
                if kind in counters:
                    counters[kind] += 1
                    numbers[m.group(1)] = str(counters[kind])
    return numbers


def resolve_refs(text: str, numbers: dict[str, str]) -> str:
    def repl(m):
        label = m.group(1)
        if label not in numbers:
            raise KeyError(f"unknown reference {label}")
        kind = label.split(":")[0]
        word = {"fig": "Figure", "tab": "Table", "eq": "Equation", "sec": "Section"}[kind]
        n = numbers[label]
        return f"{word} ({n})" if kind == "eq" else f"{word} {n}"
    return REF_RE.sub(repl, text)


# --------------------------------------------------------------------------- docx helpers
def set_cell_shading(cell, hex_fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:val"), "clear")
    shade.set(qn("w:color"), "auto")
    shade.set(qn("w:fill"), hex_fill)
    props.append(shade)


def add_field(paragraph, instruction: str, placeholder: str = "") -> None:
    """Insert a Word field such as a table of contents or a page number."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, text, end):
        run._r.append(el)


def add_inline(paragraph, text: str, size: float | None = None, italic_all: bool = False) -> None:
    """Add text with **bold**, *italic* and `code` spans."""
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = paragraph.add_run(tok[2:-2])
            run.bold = True
        elif tok.startswith("`") and tok.endswith("`"):
            run = paragraph.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt((size or 11) - 1)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            run = paragraph.add_run(tok[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(tok)
        if italic_all:
            run.italic = True
        if size and not (tok.startswith("`") and tok.endswith("`")):
            run.font.size = Pt(size)


def render_equation(latex: str, fontsize: int = 13) -> io.BytesIO:
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, format="png", bbox_inches="tight", pad_inches=0.03, transparent=False, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------- document
def setup_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    pf = normal.paragraph_format
    pf.space_after = Pt(5)
    pf.line_spacing = 1.15
    for name, size, before, after in (("Heading 1", 17, 0, 14), ("Heading 2", 13.5, 14, 6), ("Heading 3", 12, 10, 4)):
        st = doc.styles[name]
        st.font.name = BODY_FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = INK
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Cm(2.3)
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.2)


def title_page(doc: Document, front: dict) -> None:
    for _ in range(0 if COMPACT else 4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(front["title"])
    run.bold = True
    run.font.size = Pt(20)
    if front.get("subtitle"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(front["subtitle"])
        run.font.size = Pt(13)
        run.italic = True
    for _ in range(1 if COMPACT else 3):
        doc.add_paragraph()
    for line in front["lines"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0) if COMPACT else p.paragraph_format.space_after
        add_inline(p, line, size=11 if COMPACT else 12)
    if COMPACT:
        return
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    h = doc.add_paragraph()
    run = h.add_run("Contents")
    run.bold = True
    run.font.size = Pt(17)
    add_field(doc.add_paragraph(), 'TOC \\o "1-1" \\h \\z \\u',
              "Right-click here and choose Update Field to build the table of contents.")


def add_table(doc: Document, rows: list[list[str]], caption: str | None, numbers: dict[str, str]) -> None:
    if caption:
        m = LABEL_RE.search(caption)
        label = m.group(1) if m else None
        text = LABEL_RE.sub("", caption).strip()
        p = doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)
        run = p.add_run(f"Table {numbers[label]}. " if label else "Table. ")
        run.bold = True
        run.font.size = Pt(10.5)
        add_inline(p, resolve_refs(text, numbers), size=10.5)
    header, body = rows[0], rows[1:]
    aligns = None
    if body and all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in body[0]):
        aligns = [("right" if c.strip().endswith(":") and not c.strip().startswith(":") else "left") for c in body[0]]
        body = body[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, text in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        add_inline(cell.paragraphs[0], resolve_refs(text.strip(), numbers), size=9.5)
        for run in cell.paragraphs[0].runs:
            run.bold = True
        set_cell_shading(cell, "E9EEF2")
    for i, row in enumerate(body, start=1):
        for j, text in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, resolve_refs(text.strip(), numbers), size=9.5)
            if aligns and aligns[j] == "right":
                par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for row in table.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.line_spacing = 1.0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc: Document, caption: str, path: str, label: str | None, numbers: dict[str, str], width: float) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT / path), width=Inches(width * FIGURE_SCALE))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_after = Pt(10)
    run = cap.add_run(f"Figure {numbers[label]}. " if label else "Figure. ")
    run.bold = True
    run.font.size = Pt(10.5)
    add_inline(cap, resolve_refs(caption, numbers), size=10.5)


def add_equation(doc: Document, latex: str, label: str | None, numbers: dict[str, str]) -> None:
    """Equation image centred on the text block, its number flush right on the same line."""
    from docx.enum.text import WD_TAB_ALIGNMENT

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_together = True
    width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    p.paragraph_format.tab_stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(int(width), WD_TAB_ALIGNMENT.RIGHT)
    image = render_equation(latex)
    p.add_run("\t")
    p.add_run().add_picture(image, height=Cm(1.0) if ("frac" in latex or "max" in latex or "sum" in latex) else Cm(0.62))
    p.add_run(f"\t({numbers[label]})" if label else "\t").font.size = Pt(11)


def build() -> Path:
    front = json.loads((HERE / "front.json").read_text(encoding="utf-8"))
    files = sorted(CHAPTERS.glob("*.md"))
    contents = [f.read_text(encoding="utf-8").splitlines() for f in files]
    numbers = collect_labels(contents)

    doc = Document()
    setup_styles(doc)
    title_page(doc, front)

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(footer, "PAGE", "1")

    chapter = 0
    for lines in contents:
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line.strip():
                i += 1
                continue
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                m = LABEL_RE.search(line)
                text = LABEL_RE.sub("", line[level:]).strip()
                if level == 1:
                    if not COMPACT:
                        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                    numbered = not is_unnumbered(text)
                    if numbered:
                        chapter += 1
                        text = f"{chapter}  {text}"
                doc.add_heading(text, level=level)
                i += 1
                continue
            if line.startswith("Table:"):
                caption = line[len("Table:"):].strip()
                i += 1
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                add_table(doc, rows, caption, numbers)
                continue
            if line.strip().startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                add_table(doc, rows, None, numbers)
                continue
            fm = re.match(r"!\[(.*)\]\((.+?)\)(\{[^}]*\})?\s*$", line.strip())
            if fm:
                attrs = fm.group(3) or ""
                lm = LABEL_RE.search(attrs)
                wm = re.search(r"width=([\d.]+)", attrs)
                add_figure(doc, fm.group(1), fm.group(2), lm.group(1) if lm else None, numbers,
                           float(wm.group(1)) if wm else 6.0)
                i += 1
                continue
            em = re.match(r"\$\$(.+)\$\$\s*(\{#eq:[\w-]+\})?\s*$", line.strip())
            if em:
                lm = LABEL_RE.search(em.group(2) or "")
                add_equation(doc, em.group(1).strip(), lm.group(1) if lm else None, numbers)
                i += 1
                continue
            bm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
            if bm:
                style = "List Number" if bm.group(2)[0].isdigit() else "List Bullet"
                if len(bm.group(1)) >= 2:
                    style += " 2"
                text = bm.group(3)
                i += 1
                while i < len(lines) and lines[i].startswith("    ") and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i]):
                    text += " " + lines[i].strip()
                    i += 1
                p = doc.add_paragraph(style=style)
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, resolve_refs(text, numbers))
                continue
            # ordinary paragraph: join until a blank line or a block start
            text = line.strip()
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||Table:|!\[|\$\$|\s*[-*]\s|\s*\d+\.\s)", lines[i]):
                text += " " + lines[i].strip()
                i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(p, resolve_refs(text, numbers))

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    if "--paper" in sys.argv:
        COMPACT = True
        HERE = ROOT / "paper"
        CHAPTERS = HERE / "chapters"
        OUT = HERE / "TalentScout_v2_Paper.docx"
    out = build()
    print(f"wrote {out.relative_to(ROOT)}")
    sys.exit(0)
