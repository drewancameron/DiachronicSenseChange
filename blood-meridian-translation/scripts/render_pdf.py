#!/usr/bin/env python3
"""
Render translated passages as PDF via WeasyPrint.

Uses the same HTML output as the website (render_passage.py),
which WeasyPrint renders with CSS grid layout — glosses sit
in a column next to each paragraph, footnotes at page bottom.

Usage:
  python3 scripts/render_pdf.py          # generate PDF from HTML
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def main():
    import weasyprint

    html_path = OUTPUT / "blood_meridian.html"
    pdf_path = OUTPUT / "blood_meridian.pdf"

    if not html_path.exists():
        print(f"HTML not found at {html_path} — run render_passage.py first")
        sys.exit(1)

    doc = weasyprint.HTML(filename=str(html_path))
    doc.write_pdf(str(pdf_path))
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
