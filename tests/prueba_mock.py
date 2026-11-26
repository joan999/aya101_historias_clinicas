"""Prueba de código sin GPU: corre la app con un T5 de juguete (pesos aleatorios).

Las respuestas del modelo no tienen sentido. Solo revisa que el flujo completo
(entrada, generate, atención cruzada, tabla, Q/K/V) no falle antes de llevarlo a Colab.

Ejecutar desde la carpeta raíz del repositorio:  python tests/prueba_mock.py
"""

import os
import sys

os.environ["MODELO_ID"] = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"

from streamlit.testing.v1 import AppTest

app = AppTest.from_file("../app.py", default_timeout=600)
app.run()
assert not app.exception, app.exception

for estrategia in ["Una pregunta por variable", "Un solo prompt con JSON", "Ambas (comparar)"]:
    app.radio[1].set_value(estrategia)
    app.button[0].click()
    app.run()
    assert not app.exception, app.exception
    print(estrategia, "->", [h.value for h in app.header])
    # Cambia de capa en el mapa de atención
    app.selectbox(key="capa_mapa").set_value(1)
    app.run()
    assert not app.exception, app.exception
    # Sección 5: Q, K y V en los tres tipos de atención, en otra capa y otra cabeza
    for tipo in app.selectbox(key="tipo_qkv").options:
        app.selectbox(key="tipo_qkv").set_value(tipo)
        app.selectbox(key="capa_qkv").set_value(2)
        app.selectbox(key="cabeza_qkv").set_value(2)
        app.run()
        assert not app.exception, app.exception
        comprobacion = [m.value for m in app.markdown if "Comprobación" in m.value]
        print("  ", tipo, "|", comprobacion[0] if comprobacion else "sin comprobación")

# Entrada por texto pegado
app.radio[0].set_value("Pegar texto")
app.run()
app.text_area[0].input("Paciente con HTA. TA 140/90 mmHg.")
app.run()
app.button[0].click()
app.run()
assert not app.exception, app.exception
print("tabla:\n", app.dataframe[0].value.head(3).to_string())
print("OK")
sys.exit(0)
