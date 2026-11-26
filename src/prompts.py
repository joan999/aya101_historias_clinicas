"""Textos que recibe el encoder en cada estrategia."""

from src.config import VARIABLES


def prompt_por_variable(historia, pregunta, tipo, variante="original"):
    """variante="negacion" cambia la instrucción de las preguntas Sí/No para separar
    «No» (la historia lo niega) de «No registrado» (la historia no lo menciona)."""
    if tipo == "si_no" and variante == "negacion":
        instruccion = ("Responde Sí si la historia lo afirma y No si la historia lo niega o dice lo contrario. "
                       "Si la historia no dice nada al respecto, responde: No registrado.")
    else:
        formato = "Responde solo Sí o No." if tipo == "si_no" else "Responde solo con el número, sin unidades."
        instruccion = f"{formato} Si el dato no aparece, responde: No registrado."
    return (
        f"Lee la historia clínica y responde la pregunta. {instruccion}\n\n"
        f"Historia clínica:\n{historia}\n\n"
        f"Pregunta: {pregunta}\nRespuesta:"
    )


def prompt_json(historia):
    claves = "\n".join(f"- {clave}: {pregunta}" for clave, _, pregunta in VARIABLES)
    return (
        "Extrae de la historia clínica los datos de la lista y responde solo con un JSON. "
        'Usa "Sí" o "No" en las preguntas de sí o no, números sin unidades en las medidas '
        "y null si el dato no aparece.\n\n"
        f"Historia clínica:\n{historia}\n\n"
        f"Claves del JSON:\n{claves}\n\nJSON:"
    )
