"""Colores y mapas de calor que comparten la app y los notebooks.

Los colores salen de una paleta revisada para daltonismo y contraste:
- dos estrategias: azul y naranja;
- magnitudes de 0 a 1 (pesos de atención): un solo tono, de claro a oscuro;
- valores con signo (puntajes, sesgo): azul y rojo, con gris en el cero.
"""

from matplotlib.colors import CenteredNorm, LinearSegmentedColormap

COLOR_ESTRATEGIA = {"Una pregunta por variable": "#2a78d6", "Un solo prompt con JSON": "#eb6834"}
# Orden fijo para comparar hasta tres series (azul, naranja, aguamarina), revisado con el validador de la skill
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXTO_SECUNDARIO = "#52514e"
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("azul_rojo", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#b8302f"])


def mapa_calor(eje, figura, matriz, titulo, tokens_filas, tokens_columnas, con_signo):
    """Dibuja una matriz de atención: filas = consultas (Q), columnas = claves (K)."""
    datos = matriz.numpy() if hasattr(matriz, "numpy") else matriz
    if con_signo:
        imagen = eje.imshow(datos, aspect="auto", cmap=DIVERGENTE, norm=CenteredNorm(vcenter=0))
    else:
        imagen = eje.imshow(datos, aspect="auto", cmap=SECUENCIAL, vmin=0)
    eje.set_title(titulo, fontsize=9, loc="left")
    eje.set_xlabel("claves (K)", fontsize=8, color=TEXTO_SECUNDARIO)
    eje.set_ylabel("consultas (Q)", fontsize=8, color=TEXTO_SECUNDARIO)
    # Con muchos tokens las etiquetas no se leen; se muestran solo si son pocos
    if len(tokens_filas) <= 40:
        eje.set_yticks(range(len(tokens_filas)), tokens_filas, fontsize=6)
    if len(tokens_columnas) <= 40:
        eje.set_xticks(range(len(tokens_columnas)), tokens_columnas, fontsize=6, rotation=90)
    eje.tick_params(colors=TEXTO_SECUNDARIO, labelsize=6)
    for lado in eje.spines.values():
        lado.set_visible(False)
    barra = figura.colorbar(imagen, ax=eje, shrink=0.8)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=6, colors=TEXTO_SECUNDARIO)
    return imagen
