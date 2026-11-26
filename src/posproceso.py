"""Convierte el texto que genera el modelo en valores comparables."""

import json
import re

from src.config import TIPOS


def normalizar(texto, tipo):
    """Pasa la respuesta del modelo a un valor: 'Sí', 'No', un número o None."""
    t = str(texto).strip().lower()
    if not t or "no registr" in t or t in ("null", "none"):
        return None
    if tipo == "si_no":
        if t.startswith(("sí", "si")):
            return "Sí"
        if t.startswith("no"):
            return "No"
        return None
    # Un número suelto: no toma dígitos pegados a letras, como el «1» de «HbA1c» o el «2» de «DM2»
    numero = re.search(r"(?<!\w)-?\d+(?:[.,]\d+)?", t)
    return float(numero.group().replace(",", ".")) if numero else None


def leer_json(texto):
    """Lee la salida de la estrategia JSON y devuelve un valor por variable.

    Primero intenta leerla como JSON. Si no se puede, toma el texto que hay entre
    cada clave y la siguiente, porque Aya-101 a veces escribe "clave: valor" sin
    llaves ni comas (por ejemplo "talla_m: 1,60 m creatinina_mg_dl: 0,9 mg/dL").
    """
    try:
        datos = json.loads(texto[texto.find("{"): texto.rfind("}") + 1])
    except ValueError:
        datos = {}
    if not isinstance(datos, dict) or not datos:
        datos = {}
        claves = "|".join(re.escape(clave) for clave in TIPOS)
        marcas = list(re.finditer(rf'"?({claves})"?\s*:', texto))
        for actual, siguiente in zip(marcas, marcas[1:] + [None]):
            fin = siguiente.start() if siguiente else len(texto)
            # Si el modelo repite una clave, vale la primera vez que aparece
            datos.setdefault(actual.group(1), texto[actual.end():fin].strip().strip(",{}\"' "))
    return {clave: normalizar("" if datos.get(clave) is None else datos.get(clave), TIPOS[clave]) for clave in TIPOS}


def coincide(valor, esperado):
    if isinstance(valor, float) and isinstance(esperado, (int, float)):
        return abs(valor - esperado) < 1e-6
    return valor == esperado
