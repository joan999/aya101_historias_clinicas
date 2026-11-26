"""Carga de Aya-101 (o de un modelo de prueba sin GPU)."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


def cargar_modelo(modelo_id):
    """Devuelve (tokenizador, modelo).

    Se usa atención "eager" porque la de por defecto (sdpa) no devuelve los pesos de atención.
    """
    tokenizador = AutoTokenizer.from_pretrained(modelo_id)
    if torch.cuda.is_available():
        # En la T4 de Colab solo cabe en 4 bits; los cálculos van en bf16
        cuantizacion = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        modelo = AutoModelForSeq2SeqLM.from_pretrained(
            modelo_id, quantization_config=cuantizacion, dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
        )
    else:
        # Sin GPU solo sirve para modelos pequeños de prueba
        modelo = AutoModelForSeq2SeqLM.from_pretrained(modelo_id, attn_implementation="eager")
    modelo.eval()
    return tokenizador, modelo
