"""Lectura de la historia clínica cuando llega en PDF."""

import pymupdf


def leer_pdf(contenido):
    """Recibe los bytes del PDF y devuelve su texto."""
    with pymupdf.open(stream=contenido, filetype="pdf") as doc:
        return "\n".join(pagina.get_text() for pagina in doc).strip()
