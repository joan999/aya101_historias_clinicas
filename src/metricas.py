"""Métricas de desempeño de la extracción.

Cada casilla (historia × variable) se compara con su valor esperado:

  el modelo...                el valor esperado...   cuenta como
  da un valor correcto        existe                 VP
  da un valor equivocado      existe                 FP y FN (llenó mal y no dio el valor real)
  da un valor                 no existe              FP (lo inventó)
  no da valor                 existe                 FN
  no da valor                 no existe              VN (no entra en precisión, recall ni F1)

Precisión = VP / (VP + FP): de lo que llena el modelo, cuánto está bien.
Recall    = VP / (VP + FN): de lo que está en la historia, cuánto encuentra.
F1        = media armónica de las dos.
"""

import json
import math
import random

from src.posproceso import coincide


def clasificar(valor, esperado):
    """correcto, valor distinto, omitido o inventado."""
    if coincide(valor, esperado):
        return "correcto"
    if valor is None:
        return "omitido"
    if esperado is None:
        return "inventado"
    return "valor distinto"


def conteos(valor, esperado):
    """Devuelve (VP, FP, FN) de una casilla."""
    if valor is None:
        return (0, 0, 1 if esperado is not None else 0)
    if coincide(valor, esperado):
        return (1, 0, 0)
    return (0, 1, 1 if esperado is not None else 0)


def precision_recall_f1(pares):
    """pares: lista de (valor, esperado). Suma las casillas (promedio micro)."""
    vp = fp = fn = 0
    for valor, esperado in pares:
        a, b, c = conteos(valor, esperado)
        vp, fp, fn = vp + a, fp + b, fn + c
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"VP": vp, "FP": fp, "FN": fn, "precisión": precision, "recall": recall, "F1": f1}


def intervalo_wilson(aciertos, total, z=1.96):
    """Intervalo de confianza del 95 % para una proporción (método de Wilson)."""
    if total == 0:
        return (0.0, 0.0)
    p = aciertos / total
    centro = (p + z * z / (2 * total)) / (1 + z * z / total)
    margen = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def intervalo_bootstrap_f1(pares_por_caso, repeticiones=2000, semilla=42):
    """Intervalo del 95 % para el F1 micro, remuestreando historias completas.

    Se remuestrean historias y no casillas sueltas, porque las 10 casillas de una
    misma historia no son independientes entre sí.
    """
    azar = random.Random(semilla)
    casos = list(pares_por_caso.values())
    valores = []
    for _ in range(repeticiones):
        muestra = [par for caso in azar.choices(casos, k=len(casos)) for par in caso]
        valores.append(precision_recall_f1(muestra)["F1"])
    valores.sort()
    return (valores[int(0.025 * repeticiones)], valores[int(0.975 * repeticiones) - 1])


def es_json_valido(texto):
    """True si la salida se puede leer como un objeto JSON sin el lector de respaldo."""
    try:
        datos = json.loads(texto[texto.find("{"): texto.rfind("}") + 1])
    except ValueError:
        return False
    return isinstance(datos, dict) and bool(datos)
