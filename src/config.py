"""Modelo, largo de las respuestas y variables que se extraen."""

import os

MODELO_ID = os.environ.get("MODELO_ID", "CohereLabs/aya-101")
MAX_TOKENS_VARIABLE = 40   # con 20 se cortaba la respuesta de HbA1c antes del número (corrida 1)
MAX_TOKENS_JSON = 200      # el JSON con las 10 claves
NO_REPEAT_JSON = 10        # en el JSON no se puede repetir un grupo de 10 tokens; así se cortan los ciclos «Sí, no.»

# Variables del subconjunto ilustrativo: (clave, tipo, pregunta para el modelo)
VARIABLES = [
    ("diagnostico_hta", "si_no", "¿El paciente tiene diagnóstico de hipertensión arterial?"),
    ("diagnostico_dm", "si_no", "¿El paciente tiene diagnóstico de diabetes mellitus?"),
    ("diagnostico_erc", "si_no", "¿El paciente tiene diagnóstico de enfermedad renal crónica?"),
    ("tas_mmhg", "numero", "¿Cuál es la presión arterial sistólica en mmHg?"),
    ("tad_mmhg", "numero", "¿Cuál es la presión arterial diastólica en mmHg?"),
    ("peso_kg", "numero", "¿Cuál es el peso del paciente en kilogramos?"),
    ("talla_m", "numero", "¿Cuál es la talla del paciente en metros?"),
    ("creatinina_mg_dl", "numero", "¿Cuál es el valor de creatinina en mg/dL?"),
    ("hba1c_pct", "numero", "¿Cuál es el valor de hemoglobina glicosilada (HbA1c) en %?"),
    ("usa_ieca_ara2", "si_no", "¿El paciente recibe actualmente un IECA o un ARA-II (por ejemplo enalapril o losartán)?"),
]
TIPOS = {clave: tipo for clave, tipo, _ in VARIABLES}
