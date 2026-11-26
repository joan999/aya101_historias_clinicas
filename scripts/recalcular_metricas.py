"""Recalcula las métricas desde el texto que generó el modelo, con el lector actual.

No usa GPU: vuelve a leer la columna «respuesta» de un evaluacion_detalle.csv y
compara con ejemplos_hc/esperados.json.

Ejecutar desde la carpeta raíz del repositorio:
    python scripts/recalcular_metricas.py doc/metricas/evaluacion_detalle.csv doc/metricas/recalculo_corrida1
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TIPOS
from src.metricas import (clasificar, es_json_valido, intervalo_bootstrap_f1, intervalo_wilson,
                          precision_recall_f1)
from src.posproceso import leer_json, normalizar


def recalcular(detalle, esperados):
    """Vuelve a leer cada respuesta y la clasifica contra el valor esperado."""
    filas = []
    leidos_json = {}
    for fila in detalle.to_dict("records"):
        clave, caso = fila["variable"], fila["caso"]
        if "JSON" in fila["estrategia"]:
            # En la estrategia JSON todas las variables comparten la misma respuesta
            llave = (fila["estrategia"], caso)
            if llave not in leidos_json:
                leidos_json[llave] = leer_json(fila["respuesta"])
            valor = leidos_json[llave][clave]
        else:
            valor = normalizar(fila["respuesta"], TIPOS[clave])
        esperado = esperados[caso][clave]
        filas.append({**fila, "valor": valor, "esperado": esperado, "resultado": clasificar(valor, esperado)})
    return filas


def resumen(filas):
    """Exactitud y precisión, recall y F1 por estrategia, con sus intervalos."""
    salida = []
    for estrategia in dict.fromkeys(f["estrategia"] for f in filas):
        propias = [f for f in filas if f["estrategia"] == estrategia]
        aciertos = sum(f["resultado"] == "correcto" for f in propias)
        bajo_e, alto_e = intervalo_wilson(aciertos, len(propias))
        por_caso = {}
        for f in propias:
            por_caso.setdefault(f["caso"], []).append((f["valor"], f["esperado"]))
        m = precision_recall_f1([(f["valor"], f["esperado"]) for f in propias])
        bajo_f, alto_f = intervalo_bootstrap_f1(por_caso, repeticiones=2000, semilla=42)
        conteo = pd.Series([f["resultado"] for f in propias]).value_counts()
        salida.append({
            "estrategia": estrategia, "aciertos": aciertos, "total": len(propias),
            "exactitud": round(aciertos / len(propias), 3), "IC 95 % exactitud": f"{bajo_e:.3f} – {alto_e:.3f}",
            "VP": m["VP"], "FP": m["FP"], "FN": m["FN"], "precisión": round(m["precisión"], 3),
            "recall": round(m["recall"], 3), "F1": round(m["F1"], 3), "IC 95 % F1": f"{bajo_f:.3f} – {alto_f:.3f}",
            **{f"n_{k}": int(conteo.get(k, 0)) for k in ["correcto", "valor distinto", "omitido", "inventado"]},
        })
    return pd.DataFrame(salida)


if __name__ == "__main__":
    origen, destino = Path(sys.argv[1]), Path(sys.argv[2])
    detalle = pd.read_csv(origen, dtype=str, keep_default_na=False)
    esperados = json.loads(Path("ejemplos_hc/esperados.json").read_text(encoding="utf-8"))
    filas = recalcular(detalle, esperados)
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(destino / "evaluacion_detalle.csv", index=False)
    tabla = resumen(filas)
    tabla.to_csv(destino / "resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(tabla.to_string(index=False))
    respuestas_json = detalle[detalle["estrategia"].str.contains("JSON")].drop_duplicates(["estrategia", "caso"])["respuesta"]
    if len(respuestas_json):
        print(f"JSON válido: {sum(es_json_valido(r) for r in respuestas_json)} de {len(respuestas_json)}")
