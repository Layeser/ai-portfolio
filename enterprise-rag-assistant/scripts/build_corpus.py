"""Render the Markdown policy sources of the fictional company *Atelier Nord* to PDF.

The RAG pipeline ingests PDFs (the realistic case); Markdown is only the
authoring format. File names follow `<department>__<name>.md`.
"""

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "corpus" / "sources"
PDF_DIR = ROOT / "corpus" / "pdf"

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
TITLE = ParagraphStyle("t", fontName="DejaVuBold", fontSize=16, leading=20, spaceAfter=14)
HEADING = ParagraphStyle("h", fontName="DejaVuBold", fontSize=12, leading=16, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("b", fontName="DejaVu", fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=8)


def render(md_path: Path) -> Path:
    department, name = md_path.stem.split("__")
    out = PDF_DIR / department / f"{name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    story = []
    for block in md_path.read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        if block.startswith("# "):
            story.append(Paragraph(block[2:], TITLE))
        elif block.startswith("## "):
            story.append(Paragraph(block[3:], HEADING))
        elif block:
            story.append(Paragraph(block.replace("\n", " "), BODY))
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)
    return out


if __name__ == "__main__":
    for md in sorted(SOURCES.glob("*.md")):
        print("wrote", render(md).relative_to(ROOT))
