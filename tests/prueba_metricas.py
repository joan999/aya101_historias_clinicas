"""Prueba de las métricas con ejemplos hechos a mano.

Ejecutar desde la carpeta raíz del repositorio:  python tests/prueba_metricas.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metricas import (clasificar, conteos, es_json_valido, intervalo_bootstrap_f1,
                          intervalo_wilson, precision_recall_f1)

# Las cinco situaciones de la tabla del módulo
assert conteos(138.0, 138) == (1, 0, 0) and clasificar(138.0, 138) == "correcto"
assert conteos(1.0, 1.6) == (0, 1, 1) and clasificar(1.0, 1.6) == "valor distinto"
assert conteos(7.0, None) == (0, 1, 0) and clasificar(7.0, None) == "inventado"
assert conteos(None, 96) == (0, 0, 1) and clasificar(None, 96) == "omitido"
assert conteos(None, None) == (0, 0, 0) and clasificar(None, None) == "correcto"

# Caso 2 de la primera prueba en Colab, con el lector corregido: 8 correctos, 1 omitido (TAD) y 1 inventado (HbA1c)
pares = [("Sí", "Sí"), ("No", "No"), ("No", "No"), (152.0, 152), (None, 96), (82.0, 82),
         (1.6, 1.6), (0.9, 0.9), (7.0, None), ("Sí", "Sí")]
m = precision_recall_f1(pares)
assert (m["VP"], m["FP"], m["FN"]) == (8, 1, 1), m
assert abs(m["precisión"] - 8 / 9) < 1e-9 and abs(m["recall"] - 8 / 9) < 1e-9 and abs(m["F1"] - 8 / 9) < 1e-9

# Wilson: valor conocido para 8 de 10 (0,4902 a 0,9433)
bajo, alto = intervalo_wilson(8, 10)
assert abs(bajo - 0.4902) < 1e-3 and abs(alto - 0.9433) < 1e-3, (bajo, alto)
assert intervalo_wilson(10, 10)[1] == 1.0

# Bootstrap: con todas las historias perfectas el intervalo es (1, 1); además es reproducible
perfectas = {f"caso_{i}": [(1.0, 1)] * 10 for i in range(15)}
assert intervalo_bootstrap_f1(perfectas, repeticiones=200) == (1.0, 1.0)
mixtas = {"a": pares, "b": [("Sí", "Sí")] * 10, "c": [(None, 1)] * 10}
assert intervalo_bootstrap_f1(mixtas, 300) == intervalo_bootstrap_f1(mixtas, 300)

# JSON válido: el texto de esa primera prueba no lo es
assert not es_json_valido("diagnostico_hta: Sí diagnostico_dm: No")
assert es_json_valido('Respuesta: {"tas_mmhg": 138, "hba1c_pct": null}')
assert not es_json_valido("{}")

print("OK")
