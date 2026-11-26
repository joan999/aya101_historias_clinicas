"""Demo de inferencia con Aya-101 (Transformer encoder-decoder).

Lee una historia clínica (texto o PDF), le pide al modelo 10 variables de la
cohorte de riesgo cardiovascular (RCV) y muestra la salida, la atención cruzada
y los tensores Q, K y V de una capa.
La lógica está en src/; este archivo solo arma la interfaz.

Ejecutar:  streamlit run app.py   (necesita GPU, ver notebooks/despliegue_colab.ipynb)
"""

import json
import time
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import MAX_TOKENS_JSON, MAX_TOKENS_VARIABLE, MODELO_ID, NO_REPEAT_JSON, TIPOS, VARIABLES
from src.entrada import leer_pdf
from src.graficos import mapa_calor
from src.inferencia import inferir
from src.modelo import cargar_modelo
from src.posproceso import coincide, leer_json, normalizar
from src.prompts import prompt_json, prompt_por_variable
from src.qkv import TIPOS_ATENCION, capturar_qkv

CARPETA_EJEMPLOS = Path(__file__).parent / "ejemplos_hc"


@st.cache_resource(show_spinner=False)
def modelo_en_cache(modelo_id):
    """Carga el modelo una sola vez por sesión del servidor."""
    return cargar_modelo(modelo_id)


# ---------- Interfaz ----------

st.set_page_config(page_title="Aya-101 RCV", layout="wide")
st.title("Extracción de variables RCV con Aya-101")
st.caption(f"Trabajo final de Procesamiento de Datos Secuenciales. Modelo: {MODELO_ID} (encoder-decoder). "
           "Las historias de ejemplo son sintéticas, con datos ficticios.")

with st.spinner("Cargando el modelo (la primera vez tarda varios minutos)..."):
    tokenizador, modelo = modelo_en_cache(MODELO_ID)

# 1. Entrada
st.header("1. Entrada")
fuente = st.radio("Fuente", ["Ejemplo sintético", "Pegar texto", "Subir PDF"], horizontal=True)
texto_base, esperados = "", None
if fuente == "Ejemplo sintético":
    caso = st.selectbox("Ejemplo", sorted((p.name for p in CARPETA_EJEMPLOS.glob("caso_*.txt")), key=lambda n: int(n[5:-4])))
    texto_base = (CARPETA_EJEMPLOS / caso).read_text(encoding="utf-8")
    esperados = json.loads((CARPETA_EJEMPLOS / "esperados.json").read_text(encoding="utf-8"))[caso]
elif fuente == "Subir PDF":
    pdf = st.file_uploader("Historia clínica en PDF", type="pdf")
    if pdf:
        texto_base = leer_pdf(pdf.read())
historia = st.text_area("Texto que entra al modelo", value=texto_base, height=220)

estrategia = st.radio(
    "Estrategia de extracción",
    ["Una pregunta por variable", "Un solo prompt con JSON", "Ambas (comparar)"],
    horizontal=True,
)

if st.button("Ejecutar inferencia", type="primary", disabled=not historia.strip()):
    historia = historia.strip()
    corridas = {}
    if estrategia != "Un solo prompt con JSON":
        inicio = time.time()
        prompts = [prompt_por_variable(historia, pregunta, tipo) for _, tipo, pregunta in VARIABLES]
        salida = inferir(tokenizador, modelo, prompts, historia, MAX_TOKENS_VARIABLE)
        for (clave, _, _), r in zip(VARIABLES, salida):
            r["nombre"] = clave
        corridas["Una pregunta por variable"] = {"salida": salida, "segundos": time.time() - inicio}
    if estrategia != "Una pregunta por variable":
        inicio = time.time()
        salida = inferir(tokenizador, modelo, [prompt_json(historia)], historia, MAX_TOKENS_JSON,
                         no_repeat_ngram_size=NO_REPEAT_JSON)
        salida[0]["nombre"] = "JSON completo"
        corridas["Un solo prompt con JSON"] = {"salida": salida, "segundos": time.time() - inicio}
    st.session_state["corridas"] = corridas
    st.session_state["esperados"] = esperados

corridas = st.session_state.get("corridas")
if corridas:
    esperados = st.session_state.get("esperados")

    # 2. Lo que recibe el modelo
    st.header("2. Prompt que entra al encoder")
    for nombre, corrida in corridas.items():
        with st.expander(f"{nombre}: {len(corrida['salida'])} prompt(s)"):
            for r in corrida["salida"]:
                st.markdown(f"**{r['nombre']}**")
                st.code(r["prompt"], language=None)

    # 3. Salida
    st.header("3. Salida")
    tabla = pd.DataFrame({"variable": list(TIPOS)})
    for nombre, corrida in corridas.items():
        if nombre == "Una pregunta por variable":
            tabla["respuesta (por variable)"] = [r["texto"] for r in corrida["salida"]]
            tabla["valor (por variable)"] = [normalizar(r["texto"], TIPOS[r["nombre"]]) for r in corrida["salida"]]
        else:
            st.markdown("Texto generado por el decoder con la estrategia JSON:")
            st.code(corrida["salida"][0]["texto"], language=None)
            valores = leer_json(corrida["salida"][0]["texto"])
            tabla["valor (JSON)"] = [valores[c] for c in TIPOS]
        st.caption(f"{nombre}: {corrida['segundos']:.1f} s")
    if esperados:
        tabla["esperado"] = [esperados[c] for c in TIPOS]
        for col in [c for c in tabla.columns if c.startswith("valor")]:
            aciertos = [coincide(v, e) for v, e in zip(tabla[col], tabla["esperado"])]
            tabla[f"¿coincide? {col[6:]}"] = ["sí" if a else "no" for a in aciertos]
            st.write(f"Coincidencias {col[6:]}: {sum(aciertos)} de {len(aciertos)}")
    tabla = tabla.astype(object).map(lambda v: "(sin dato)" if v is None or pd.isna(v) else str(v))
    st.dataframe(tabla, hide_index=True)
    st.download_button("Descargar tabla (CSV)", tabla.to_csv(index=False).encode("utf-8"), "variables_rcv.csv")

    # 4. Atención cruzada
    st.header("4. Atención cruzada (decoder → historia)")
    st.caption("Cada fila es un token que generó el decoder y cada columna un token de la historia. "
               "El color es el peso de atención promediado sobre las 64 cabezas. "
               "Pasa el cursor por el mapa para ver el token.")
    c1, c2, c3 = st.columns(3)
    nombre = c1.selectbox("Estrategia", list(corridas), key="estrategia_mapa")
    salida = corridas[nombre]["salida"]
    r = salida[c2.selectbox("Salida", range(len(salida)), format_func=lambda i: salida[i]["nombre"], key="salida_mapa")]
    n_capas = r["atencion"].shape[0]
    capa = c3.selectbox("Capa del decoder", ["Promedio de todas"] + list(range(1, n_capas + 1)), key="capa_mapa")
    pesos = r["atencion"].mean(dim=0) if capa == "Promedio de todas" else r["atencion"][capa - 1]

    if pesos.numel() == 0:
        st.info("No hay tokens generados o de la historia para mostrar.")
    else:
        filas = [f"{i + 1}: {t}" for i, t in enumerate(r["tokens_salida"])]
        datos = pd.DataFrame([
            {"salida": filas[i], "posicion": j, "token": r["tokens_historia"][j], "peso": float(pesos[i, j])}
            for i in range(pesos.shape[0]) for j in range(pesos.shape[1])
        ])
        mapa = alt.Chart(datos).mark_rect().encode(
            x=alt.X("posicion:O", title="tokens de la historia", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("salida:N", sort=filas, title="tokens generados"),
            color=alt.Color("peso:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["salida", "posicion", "token", alt.Tooltip("peso:Q", format=".4f")],
        ).properties(height=max(120, 22 * len(filas)))
        st.altair_chart(mapa)

        st.markdown("Los 10 tokens de la historia con más atención (promedio de los tokens generados):")
        top = pesos.mean(dim=0)
        orden = top.argsort(descending=True)[:10].tolist()
        st.dataframe(pd.DataFrame({
            "posición": orden,
            "token": [r["tokens_historia"][j] for j in orden],
            "peso medio": [round(float(top[j]), 4) for j in orden],
        }), hide_index=True)

    # 5. Q, K y V
    st.header("5. Tensores Q, K y V")
    st.caption("Se repite el paso del modelo para la salida elegida en la sección 4 y se capturan los tensores "
               "dentro de la capa elegida. T5 no divide Q·Kᵀ por la raíz de d_kv: en su lugar suma un sesgo "
               "que depende de la distancia entre tokens.")
    d1, d2, d3 = st.columns(3)
    tipo_atencion = d1.selectbox("Atención", list(TIPOS_ATENCION), key="tipo_qkv")
    cruzada = TIPOS_ATENCION[tipo_atencion][1] == 1
    pila = getattr(modelo, TIPOS_ATENCION[tipo_atencion][0]).block
    capa_qkv = d2.selectbox("Capa", range(1, len(pila) + 1), key="capa_qkv")
    llave = (nombre, r["nombre"], tipo_atencion, capa_qkv)
    if st.session_state.get("qkv_llave") != llave:
        with st.spinner("Capturando Q, K y V..."):
            st.session_state["qkv"] = capturar_qkv(modelo, r["ids_entrada"], r["ids_decoder"], tipo_atencion, capa_qkv - 1)
            st.session_state["qkv_llave"] = llave
    c = st.session_state["qkv"]
    cabeza = d3.selectbox("Cabeza", range(1, c["cabezas"] + 1), key="cabeza_qkv")
    h = cabeza - 1

    # Qué tokens son filas (consultas) y cuáles columnas (claves)
    if tipo_atencion.startswith("Self-attention del encoder"):
        tokens_q, tokens_k = r["tokens_entrada"], r["tokens_entrada"]
    else:
        tokens_q = r["tokens_decoder"]
        tokens_k = r["tokens_entrada"] if cruzada else r["tokens_decoder"]

    def forma(t):
        return "[" + ", ".join(str(x) for x in t.shape) + "]"

    d_model, interna = c["x_q"].shape[-1], c["q"].shape[-1]
    origen_kv = "la salida del encoder" if cruzada else "la misma secuencia que Q"
    st.dataframe(pd.DataFrame([
        ("X: entrada de la capa (ya pasó por RMSNorm)", "", forma(c["x_q"])),
        ("Q = X·W_Q", f"W_Q: {d_model} × {interna}; se parte en {c['cabezas']} cabezas de {c['d_kv']}",
         f"{forma(c['q'])} → {forma(c['q_cabezas'])}"),
        ("K = X·W_K", f"K y V salen de {origen_kv}", f"{forma(c['k'])} → {forma(c['k_cabezas'])}"),
        ("V = X·W_V", "", f"{forma(c['v'])} → {forma(c['v_cabezas'])}"),
        ("Puntajes = Q·Kᵀ + sesgo", f"escala {c['escala']}: no divide por la raíz de {c['d_kv']}", forma(c["puntajes"])),
        ("Pesos = softmax(puntajes + máscara)", "", forma(c["pesos"])),
        ("Salida = pesos·V → se unen las cabezas → W_O", "", f"[1, {c['q'].shape[1]}, {d_model}]"),
    ], columns=["paso", "detalle", "forma"]), hide_index=True)

    if cruzada:
        st.write("En la atención cruzada T5 no usa sesgo de posición: el sesgo es cero.")
    elif c["tiene_sesgo_propio"]:
        st.write(f"Esta capa calcula el sesgo de posición relativa ({modelo.config.relative_attention_num_buckets} "
                 "grupos según la distancia entre tokens). Las demás capas de la misma pila lo reutilizan.")
    else:
        st.write("Esta capa reutiliza el sesgo de posición que se calculó en la capa 1 de la misma pila.")
    if tipo_atencion.startswith("Self-attention del decoder"):
        st.write("En el decoder, los puntajes del triángulo superior (tokens futuros) se anulan con la máscara "
                 "causal antes del softmax; por eso los pesos quedan en forma de triángulo.")
    st.write(f"Comprobación: el softmax de estos puntajes coincide con los pesos que calculó el modelo "
             f"(diferencia máxima {c['diferencia']:.1e}).")

    # (título, matriz, tiene signo): puntajes y sesgo pueden ser negativos; los pesos van de 0 a 1
    paneles = [("Puntajes antes del softmax", c["puntajes"][0, h], True)]
    if not cruzada:
        paneles.append(("Sesgo de posición relativa", c["sesgo"][0, h], True))
    paneles.append(("Pesos después del softmax", c["pesos"][0, h], False))
    figura, ejes = plt.subplots(1, len(paneles), figsize=(5 * len(paneles), 4))
    for eje, (titulo, matriz, con_signo) in zip(ejes, paneles):
        mapa_calor(eje, figura, matriz, f"{titulo} (cabeza {cabeza})", tokens_q, tokens_k, con_signo)
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

    # Valores reales: la última consulta y la clave a la que más atiende en esta cabeza
    i = len(tokens_q) - 1
    j = int(c["pesos"][0, h, i].argmax())
    st.markdown(f"Primeros 8 de los {c['d_kv']} valores en la cabeza {cabeza}: Q de la consulta «{tokens_q[i]}» "
                f"y K, V de la clave a la que más atiende, «{tokens_k[j]}» (peso {float(c['pesos'][0, h, i, j]):.3f}).")
    st.dataframe(pd.DataFrame({
        "Q": c["q_cabezas"][0, h, i, :8].numpy(),
        "K": c["k_cabezas"][0, h, j, :8].numpy(),
        "V": c["v_cabezas"][0, h, j, :8].numpy(),
    }).round(4).T, hide_index=False)
