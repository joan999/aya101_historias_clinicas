"""Prueba del posproceso con salidas de Aya-101.

El primer caso es el texto que generó Aya-101 en Colab para caso_2 con la
estrategia JSON (captura doc/img/app_primera_prueba.png).

Ejecutar desde la carpeta raíz del repositorio:  python tests/prueba_posproceso.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.posproceso import coincide, leer_json, normalizar

# Salida real de la primera prueba en Colab: pares "clave: valor" sin llaves ni comas
SALIDA_REAL_CASO_2 = (
    "diagnostico_hta: Sí diagnostico_dm: No diagnostico_erc: No tas_mmhg: 152/96 mmHg "
    "peso_kg: 82 kg talla_m: 1,60 m creatinina_mg_dl: 0,9 mg/dL hba1c_pct: 7,0 % usa_ieca_ara2: Sí"
)
leido = leer_json(SALIDA_REAL_CASO_2)
assert leido == {
    "diagnostico_hta": "Sí", "diagnostico_dm": "No", "diagnostico_erc": "No",
    "tas_mmhg": 152.0, "tad_mmhg": None, "peso_kg": 82.0, "talla_m": 1.6,
    "creatinina_mg_dl": 0.9, "hba1c_pct": 7.0, "usa_ieca_ara2": "Sí",
}, leido

# Contra lo esperado quedan 8 de 10: el modelo no dio la diastólica y puso una HbA1c que no está
esperados = json.loads(Path("ejemplos_hc/esperados.json").read_text(encoding="utf-8"))["caso_2.txt"]
fallos = [c for c in esperados if not coincide(leido[c], esperados[c])]
assert fallos == ["tad_mmhg", "hba1c_pct"], fallos

# JSON válido, con decimales en texto y null
assert leer_json('{"talla_m": "1,68", "hba1c_pct": null, "tas_mmhg": 138}')["talla_m"] == 1.68
assert leer_json('{"talla_m": "1,68", "hba1c_pct": null, "tas_mmhg": 138}')["hba1c_pct"] is None

# Pares con comillas y comas, y una clave repetida (vale la primera)
leido = leer_json('"peso_kg": "76", "creatinina_mg_dl": "1,15 mg/dL", "peso_kg": "80"')
assert leido["peso_kg"] == 76.0 and leido["creatinina_mg_dl"] == 1.15, leido

# Respuestas cortas de la estrategia por variable
assert normalizar("Sí", "si_no") == "Sí" and normalizar("No registrado", "si_no") is None
assert normalizar("7,4 %", "numero") == 7.4 and normalizar("No registrado", "numero") is None

# Respuesta cortada de la corrida 1 en Colab: no hay número, y el «1» de «HbA1c» no cuenta
assert normalizar("El valor de hemoglobina glicosilada (HbA1c) en % es", "numero") is None
assert normalizar("HbA1c 7,4 %", "numero") == 7.4 and normalizar("DM2, TA 138/88", "numero") == 138.0

print("OK")
