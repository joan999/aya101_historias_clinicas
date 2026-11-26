"""Crea el PDF de cada historia sintética que todavía no lo tenga.

Ejecutar desde la carpeta raíz del repositorio:  python ejemplos_hc/crear_pdfs.py
"""

from pathlib import Path

import pymupdf

for txt in sorted(Path(__file__).parent.glob("caso_*.txt")):
    pdf = txt.with_suffix(".pdf")
    if pdf.exists():
        continue
    doc = pymupdf.open()
    pagina = doc.new_page()
    pagina.insert_textbox(pymupdf.Rect(50, 50, 545, 790), txt.read_text(encoding="utf-8"), fontsize=11, fontname="helv")
    doc.save(pdf)
    print("creado", pdf.name)
