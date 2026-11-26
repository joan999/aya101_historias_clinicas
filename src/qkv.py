"""Captura Q, K y V de una capa de atención de Aya-101 (arquitectura T5).

Cómo calcula T5 la atención (transformers 5.24, models/t5/modeling_t5.py):
  X        = RMSNorm(estado oculto)               [1, n, 4096]
  Q, K, V  = X·W_Q, X·W_K, X·W_V                  [1, n, 4096] -> [1, 64 cabezas, n, 64]
             (en la atención cruzada K y V salen de la salida del encoder)
  puntajes = Q·Kᵀ + sesgo de posición relativa     sin dividir por raíz de d_kv
  pesos    = softmax(puntajes + máscara)
  salida   = pesos·V, se unen las cabezas y se multiplica por W_O
"""

import torch

# Nombre en la interfaz -> (pila, subcapa dentro del bloque, atributo)
TIPOS_ATENCION = {
    "Self-attention del encoder": ("encoder", 0, "SelfAttention"),
    "Self-attention del decoder": ("decoder", 0, "SelfAttention"),
    "Atención cruzada (decoder → encoder)": ("decoder", 1, "EncDecAttention"),
}


def modulo_atencion(modelo, tipo, capa):
    pila, subcapa, atributo = TIPOS_ATENCION[tipo]
    return getattr(getattr(modelo, pila).block[capa].layer[subcapa], atributo)


@torch.no_grad()
def capturar_qkv(modelo, ids_entrada, ids_decoder, tipo, capa):
    """Repite un paso del modelo y guarda lo que pasa dentro de una capa de atención.

    ids_entrada: tokens del prompt (encoder). ids_decoder: inicio + tokens ya generados.
    capa empieza en 0.
    """
    atencion = modulo_atencion(modelo, tipo, capa)
    datos = {}

    def guardar(nombre):
        def gancho(modulo, entradas, salida):
            datos[nombre] = salida.detach().float().cpu()
            datos["entrada_" + nombre] = entradas[0].detach().float().cpu()
        return gancho

    def antes(modulo, args, kwargs):
        mascara = kwargs.get("mask")
        datos["mascara"] = None if mascara is None else mascara.detach().float().cpu()

    def despues(modulo, args, kwargs, salida):
        _, sesgo, pesos = salida
        datos["sesgo"] = sesgo.detach().float().cpu()
        datos["pesos"] = pesos.detach().float().cpu()

    ganchos = [
        atencion.q.register_forward_hook(guardar("q")),
        atencion.k.register_forward_hook(guardar("k")),
        atencion.v.register_forward_hook(guardar("v")),
        atencion.register_forward_pre_hook(antes, with_kwargs=True),
        atencion.register_forward_hook(despues, with_kwargs=True),
    ]
    try:
        dispositivo = modelo.get_input_embeddings().weight.device
        modelo(input_ids=ids_entrada[None].to(dispositivo), decoder_input_ids=ids_decoder[None].to(dispositivo))
    finally:
        for gancho in ganchos:
            gancho.remove()

    cabezas, d_kv = atencion.n_heads, atencion.key_value_proj_dim

    def por_cabeza(t):
        return t.view(t.shape[0], t.shape[1], cabezas, d_kv).transpose(1, 2)

    q, k, v = por_cabeza(datos["q"]), por_cabeza(datos["k"]), por_cabeza(datos["v"])
    # Se rehace la cuenta del modelo para comprobar que la explicación es la correcta
    puntajes = q @ k.transpose(-1, -2) * atencion.scaling + datos["sesgo"]
    con_mascara = puntajes if datos["mascara"] is None else puntajes + datos["mascara"]
    diferencia = (con_mascara.softmax(dim=-1) - datos["pesos"]).abs().max().item()

    return {
        "x_q": datos["entrada_q"], "x_kv": datos["entrada_k"],
        "q": datos["q"], "k": datos["k"], "v": datos["v"],
        "q_cabezas": q, "k_cabezas": k, "v_cabezas": v,
        "puntajes": puntajes, "sesgo": datos["sesgo"], "pesos": datos["pesos"],
        "diferencia": diferencia, "cabezas": cabezas, "d_kv": d_kv,
        "escala": atencion.scaling, "tiene_sesgo_propio": atencion.has_relative_attention_bias,
    }
