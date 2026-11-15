import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from descargar_datos import ZONAS_CHILE
from modelo_rotacion import (
    cargar_catalogo_hortalizas,
    generar_escenarios_rotacion,
)
from geolocalizacion_datos import (
    extraer_coordenadas_mapa,
    caracterizar_parcela_automatica,
)

st.set_page_config(
    page_title="AgroSatélite NASA | Rotación de Cultivos Chile",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Iconografía visual para cada hortaliza (amigable para el productor)
ICONOS_CULTIVOS = {
    "Papa": "🥔",
    "Tomate": "🍅",
    "Pimentón": "🫑",
    "Cebolla": "🧅",
    "Lechuga": "🥬",
    "Poroto verde": "🫛",
    "Haba / Arveja": "🌿",
    "Zapallo": "🎃",
    "Zanahoria": "🥕",
    "Avena + Vicia (Cobertura)": "🌾",
}


def obtener_icono_cultivo(nombre: str) -> str:
    for clave, icono in ICONOS_CULTIVOS.items():
        if clave.lower() in nombre.lower():
            return icono
    return "🌱"


def inyectar_estilos_campo_claro():
    """CSS de alto contraste para lectura bajo luz solar directa (Tema Claro Agrícola:
    Azul Agua, Café Tierra, Verde Hojas y Azul Espacial NASA).
    """
    st.markdown(
        """
        <style>
        /* Forzar fondo claro cálido de alto contraste para lectura bajo el sol */
        .stApp {
            background-color: #FBF9F5 !important;
            color: #1C2819 !important;
        }
        /* Ocultar barra lateral para un flujo 100% guiado paso a paso en pantalla principal */
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }

        /* Tarjeta Contenedora del Tour Paso a Paso */
        .tour-step-box {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            border: 2.5px solid #C8E6C9;
            box-shadow: 0 4px 14px rgba(28, 40, 25, 0.06);
            margin-bottom: 1rem;
        }
        .step-badge {
            display: inline-block;
            background: #0B3D91;
            color: #FFFFFF !important;
            font-weight: 900;
            font-size: 0.82rem;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            margin-right: 0.5rem;
        }

        /* Banner Satelital NASA + Campo */
        .nasa-agro-banner {
            background: linear-gradient(115deg, #0B3D91 0%, #1565C0 38%, #2E7D32 78%, #5D4037 100%);
            color: #FFFFFF;
            padding: 1.35rem 1.75rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 20px rgba(11, 61, 145, 0.22);
            border: 2px solid #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .banner-left {
            display: flex;
            align-items: center;
            gap: 1.1rem;
        }
        .banner-title {
            font-size: 1.65rem;
            font-weight: 900;
            margin: 0;
            color: #FFFFFF !important;
            letter-spacing: -0.01em;
        }
        .banner-sub {
            font-size: 1.02rem;
            color: #E3F2FD !important;
            margin: 0.25rem 0 0 0;
            font-weight: 600;
        }
        .nasa-pill {
            background-color: #FC3D21;
            color: #FFFFFF !important;
            font-weight: 800;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            box-shadow: 0 3px 8px rgba(0,0,0,0.25);
            display: inline-block;
        }

        /* Títulos de sección con analogía de colores */
        .section-header {
            font-size: 1.28rem;
            font-weight: 900;
            padding: 0.55rem 1rem;
            border-radius: 12px;
            margin: 1rem 0 0.75rem 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .header-water {
            background-color: #E1F5FE;
            color: #01579B !important;
            border-left: 7px solid #0288D1;
        }
        .header-earth {
            background-color: #EFEBE9;
            color: #4E342E !important;
            border-left: 7px solid #6D4C41;
        }
        .header-leaf {
            background-color: #E8F5E9;
            color: #1B5E20 !important;
            border-left: 7px solid #2E7D32;
        }

        /* Tarjetas de Presentación para Escenarios (Poca letra, alto impacto visual) */
        .crop-card {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 1.25rem 1.3rem;
            border: 2.5px solid #D7CCC8;
            box-shadow: 0 5px 14px rgba(93, 64, 55, 0.08);
            transition: transform 0.15s ease;
            height: 100%;
        }
        .crop-card-winner {
            background: linear-gradient(180deg, #FFFFFF 0%, #F1F8E9 100%);
            border: 3.5px solid #2E7D32;
            box-shadow: 0 8px 24px rgba(46, 125, 50, 0.22);
        }
        .crop-card-danger {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFEBEE 100%);
            border: 2.5px solid #E53935;
        }
        .crop-card-water {
            background: linear-gradient(180deg, #FFFFFF 0%, #E1F5FE 100%);
            border: 3px solid #0288D1;
        }
        .crop-emoji-big {
            font-size: 3.1rem;
            line-height: 1.1;
            margin-right: 0.6rem;
        }
        .crop-title-big {
            font-size: 1.55rem;
            font-weight: 900;
            color: #1C2819 !important;
            margin: 0;
        }
        .visual-chips-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-top: 0.85rem;
            margin-bottom: 0.75rem;
        }
        .v-chip {
            border-radius: 12px;
            padding: 0.5rem 0.65rem;
            font-size: 0.88rem;
            font-weight: 700;
            display: flex;
            flex-direction: column;
        }
        .v-chip span.label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            opacity: 0.85;
        }
        .v-chip span.val {
            font-size: 1.02rem;
            font-weight: 900;
        }
        .chip-water {
            background-color: #E1F5FE;
            color: #01579B !important;
            border: 1.5px solid #81D4FA;
        }
        .chip-earth {
            background-color: #EFEBE9;
            color: #4E342E !important;
            border: 1.5px solid #BCAAA4;
        }
        .chip-leaf {
            background-color: #E8F5E9;
            color: #1B5E20 !important;
            border: 1.5px solid #A5D6A7;
        }
        .chip-sun {
            background-color: #FFF8E1;
            color: #E65100 !important;
            border: 1.5px solid #FFE082;
        }
        .progress-track {
            background-color: #E0E0E0;
            border-radius: 999px;
            height: 14px;
            width: 100%;
            overflow: hidden;
            margin: 0.45rem 0;
        }
        .short-tip {
            font-size: 0.92rem;
            font-weight: 600;
            color: #263238 !important;
            background: rgba(255,255,255,0.85);
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            border-left: 4px solid #2E7D32;
            margin-top: 0.5rem;
        }

        /* Secuencia Consolidada 12 Meses y Carta Gantt Visual */
        .sequence-master-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F1F8E9 55%, #E1F5FE 100%);
            border: 3.5px solid #2E7D32;
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(46, 125, 50, 0.16);
        }
        .seq-step-box {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            border: 2px solid #A5D6A7;
            box-shadow: 0 3px 10px rgba(0,0,0,0.04);
            height: 100%;
        }
        .gantt-wrapper {
            overflow-x: auto;
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1rem;
            border: 2.5px solid #BCAAA4;
            box-shadow: 0 4px 14px rgba(93, 64, 55, 0.08);
            margin-bottom: 1rem;
        }
        .gantt-table {
            width: 100%;
            min-width: 1080px;
            border-collapse: separate;
            border-spacing: 2px;
            font-size: 0.78rem;
        }
        .gantt-table th.month-th {
            background: #0B3D91;
            color: #FFFFFF !important;
            font-weight: 900;
            text-align: center;
            padding: 6px 2px;
            border-radius: 6px 6px 0 0;
            font-size: 0.8rem;
        }
        .gantt-table th.week-th {
            background: #E3F2FD;
            color: #0D47A1 !important;
            font-weight: 800;
            text-align: center;
            padding: 3px 1px;
            font-size: 0.68rem;
        }
        .gantt-table td.crop-label-td {
            background: #EFEBE9;
            color: #3E2723 !important;
            font-weight: 900;
            padding: 8px 10px;
            border-radius: 8px;
            white-space: nowrap;
            font-size: 0.88rem;
            border-left: 5px solid #6D4C41;
        }
        .gantt-cell {
            text-align: center;
            vertical-align: middle;
            height: 36px;
            min-width: 19px;
            border-radius: 5px;
            font-size: 0.98rem;
            line-height: 1;
        }
        .gantt-empty {
            background-color: #F5F3EF;
        }
        .gantt-siembra {
            background-color: #E1F5FE;
            border: 1px solid #4FC3F7;
        }
        .gantt-crecimiento {
            background-color: #E8F5E9;
            border: 1px solid #81C784;
        }
        .gantt-cosecha {
            background-color: #FFF8E1;
            border: 1px solid #FFB74D;
        }
        .sentinel-badge-card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            border: 2px solid #B0BEC5;
            box-shadow: 0 3px 8px rgba(0,0,0,0.04);
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def crear_medidor_variable(
    titulo: str,
    icono: str,
    valor: float,
    sufijo: str,
    min_val: float,
    max_val: float,
    color_barra: str,
    color_fondo_rango: str,
    subtitulo: str,
) -> go.Figure:
    """Crea una figura tipo Velocímetro/Termómetro (Gauge) individual para cada variable NASA."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=valor,
            number={
                "suffix": f" {sufijo}",
                "font": {"size": 26, "color": "#1C2819", "family": "Arial Black"},
            },
            title={
                "text": f"<b>{icono} {titulo}</b><br><span style='font-size:12px;color:#4E342E'>{subtitulo}</span>",
                "font": {"size": 15, "color": "#1C2819"},
            },
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#4E342E",
                    "tickfont": {"size": 11, "color": "#1C2819"},
                },
                "bar": {"color": color_barra, "thickness": 0.65},
                "bgcolor": "#FFFFFF",
                "borderwidth": 2,
                "bordercolor": "#D7CCC8",
                "steps": [
                    {"range": [min_val, min_val + (max_val - min_val) * 0.35], "color": "#F5F5F5"},
                    {
                        "range": [
                            min_val + (max_val - min_val) * 0.35,
                            min_val + (max_val - min_val) * 0.75,
                        ],
                        "color": color_fondo_rango,
                    },
                    {"range": [min_val + (max_val - min_val) * 0.75, max_val], "color": "#FFFDE7"},
                ],
            },
        )
    )
    fig.update_layout(
        height=195,
        margin=dict(l=18, r=18, t=58, b=12),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1C2819"),
    )
    return fig


def crear_figura_perfil_tierra_agua(ind: dict, zona: str) -> go.Figure:
    """Figura corte transversal de la Tierra (Café Tierra + Azul Agua):
    Muestra cuánta agua hay en cada capa del suelo según el satélite NASA.
    """
    capas = [
        "3. Suelo Profundo (GWETPROF >100cm)",
        "2. Zona de Raíces (GWETROOT 10-100cm)",
        "1. Capa Superficial (GWETTOP 0-10cm)",
    ]
    valores = [
        round(ind["humedad_prof_actual"] * 100, 1),
        round(ind["humedad_raiz_actual"] * 100, 1),
        round(ind["humedad_sup_actual"] * 100, 1),
    ]
    colores_agua = ["#01579B", "#0288D1", "#29B6F6"]

    fig = go.Figure()
    # Fondo café tierra (100% capacidad del poro)
    fig.add_trace(
        go.Bar(
            y=capas,
            x=[100, 100, 100],
            orientation="h",
            name="Capacidad del Suelo (Tierra)",
            marker=dict(color="#D7CCC8", line=dict(color="#6D4C41", width=1.5)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Relleno azul agua encima
    fig.add_trace(
        go.Bar(
            y=capas,
            x=valores,
            orientation="h",
            name="Agua Disponible (%)",
            text=[f"💧 {v}% de Agua" for v in valores],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=14, family="Arial Black"),
            marker=dict(color=colores_agua, line=dict(color="#01579B", width=2)),
        )
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=f"<b>🌱💧 Corte del Suelo desde el Satélite ({zona})</b>",
            font=dict(size=16, color="#4E342E"),
        ),
        xaxis=dict(
            title="Porcentaje de Humedad en la Tierra (%)",
            range=[0, 105],
            tickfont=dict(color="#1C2819", size=12),
            title_font=dict(color="#1C2819", size=13),
        ),
        yaxis=dict(tickfont=dict(color="#1C2819", size=13)),
        height=280,
        margin=dict(l=15, r=20, t=48, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#EFEBE9",
        showlegend=False,
    )
    return fig


def crear_radar_comparativo_escenarios(escenarios: list) -> go.Figure:
    """Gráfico Radar (Estrella) que compara visualmente las 3 opciones sin necesidad de leer texto."""
    categorias = [
        "💧 Ahorro de Agua",
        "🌱 Salud de Tierra",
        "🛡️ Corte de Plagas",
        "☀️ Energía Solar (GDD)",
        "❄️ Resistencia al Frío",
    ]

    fig = go.Figure()
    colores_esc = ["#E53935", "#0288D1", "#2E7D32"]

    for idx, esc in enumerate(escenarios):
        kc = esc["kc_promedio"]
        ahorro_pts = max(20, min(100, int((1.05 - kc) * 180)))
        aporte = esc["aporte_suelo"]
        if "Fijador" in aporte:
            suelo_pts = 96
        elif "Sanitizante" in aporte or "Cobertura" in aporte or "Descompactador" in aporte:
            suelo_pts = 85
        else:
            suelo_pts = 30

        plagas_pts = 20 if idx == 0 else 95
        gdd_pts = min(100, esc["detalles"]["cumplimiento_gdd_pct"])
        frio_pts = max(25, 100 - esc["detalles"]["dias_riesgo_helada"] * 15)

        valores = [ahorro_pts, suelo_pts, plagas_pts, gdd_pts, frio_pts]
        valores_cerrados = valores + [valores[0]]
        cats_cerradas = categorias + [categorias[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=valores_cerrados,
                theta=cats_cerradas,
                fill="toself",
                name=f"{esc['cultivo_recomendado']}",
                line=dict(color=colores_esc[idx], width=3),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>🎯 Comparación Visual de las 3 Opciones (Más grande = Mejor)</b>",
            font=dict(size=16, color="#1B5E20"),
        ),
        polar=dict(
            bgcolor="#FBF9F5",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#1C2819", size=10)),
            angularaxis=dict(tickfont=dict(color="#1C2819", size=12)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#1C2819", size=12),
        ),
        height=340,
        margin=dict(l=35, r=35, t=50, b=55),
        paper_bgcolor="#FFFFFF",
    )
    return fig


def construir_carta_gantt_html(secuencia_12m: dict) -> str:
    """Construye una Carta Gantt visual HTML/CSS responsiva de 12 Meses (Mes 1 a Mes 12)
    y 48 Semanas (S1, S2, S3, S4 por cada mes), rellenando cada bloque semanal activo
    con el emoji de la hortaliza y diferenciando Siembra, Crecimiento y Cosecha por color.
    Nota: Se devuelve sin sangría de 4 espacios para que el parser Markdown de Streamlit
    no lo interprete como bloque de código <pre><code>.
    """
    headers_meses = "".join(
        [f"<th class='month-th' colspan='4'>Mes {m}</th>" for m in range(1, 13)]
    )
    headers_semanas = "".join(
        [f"<th class='week-th'>S{s}</th>" for _ in range(12) for s in range(1, 5)]
    )

    filas_html = []
    for etapa in secuencia_12m["etapas"]:
        cultivo = etapa["cultivo"]
        icono = obtener_icono_cultivo(cultivo)
        familia = etapa["familia_botanica"]
        celdas = []
        for info_sem in etapa["semanas_48"]:
            if not info_sem["activa"]:
                celdas.append("<td class='gantt-cell gantt-empty'>·</td>")
            else:
                fase = info_sem["fase"]
                if fase == "Siembra":
                    clase_f = "gantt-siembra"
                elif fase == "Crecimiento":
                    clase_f = "gantt-crecimiento"
                else:
                    clase_f = "gantt-cosecha"
                celdas.append(
                    f"<td class='gantt-cell {clase_f}' title='{cultivo} ({fase})'>{icono}</td>"
                )

        celdas_str = "".join(celdas)
        filas_html.append(
            f"<tr>"
            f"<td class='crop-label-td'>"
            f"{etapa['orden']}. {icono} {cultivo}<br>"
            f"<span style='font-size:0.72rem; font-weight:700; color:#5D4037;'>🌿 {familia} (S{etapa['semana_inicio']}–S{etapa['semana_fin']})</span>"
            f"</td>"
            f"{celdas_str}"
            f"</tr>"
        )

    filas_str = "".join(filas_html)
    leyenda_html = (
        "<div style='display:flex; flex-wrap:wrap; gap:1rem; align-items:center; justify-content:space-between; margin-top:0.65rem; font-size:0.84rem; font-weight:700;'>"
        "<div style='display:flex; gap:0.9rem; flex-wrap:wrap;'>"
        "<span style='background:#E1F5FE; border:1.5px solid #4FC3F7; padding:3px 10px; border-radius:6px; color:#01579B;'>💧 Azul = Siembra / Trasplante</span>"
        "<span style='background:#E8F5E9; border:1.5px solid #81C784; padding:3px 10px; border-radius:6px; color:#1B5E20;'>🌱 Verde = Crecimiento y Cobertura</span>"
        "<span style='background:#FFF8E1; border:1.5px solid #FFB74D; padding:3px 10px; border-radius:6px; color:#E65100;'>🌾 Dorado = Cosecha / Incorporación</span>"
        "</div>"
        "<div style='color:#2E7D32;'>✅ <b>48 de 48 Semanas con Suelo Cubierto (0% Suelo Desnudo)</b></div>"
        "</div>"
    )
    return (
        "<div class='gantt-wrapper'>"
        "<table class='gantt-table'>"
        "<thead>"
        "<tr>"
        "<th class='month-th' rowspan='2' style='min-width:170px;'>Cultivo en Rotación<br><span style='font-size:0.7rem;font-weight:600;'>(3 Familias Distintas)</span></th>"
        f"{headers_meses}"
        "</tr>"
        f"<tr>{headers_semanas}</tr>"
        "</thead>"
        f"<tbody>{filas_str}</tbody>"
        "</table>"
        f"{leyenda_html}"
        "</div>"
    )


def construir_figura_gantt_plotly(secuencia_12m: dict) -> go.Figure:
    """Construye una matriz Carta Gantt interactiva en Plotly de 12 Meses (Mes 1 a Mes 12)
    y 48 Semanas (S1..S4 por mes) con los emojis de cada hortaliza en cada celda activa.
    """
    etapas = list(reversed(secuencia_12m["etapas"]))
    etiquetas_y = [
        f"<b>{e['orden']}. {obtener_icono_cultivo(e['cultivo'])} {e['cultivo']}</b><br><span style='font-size:11px;color:#5D4037'>{e['familia_botanica']}</span>"
        for e in etapas
    ]
    semanas_x = list(range(1, 49))

    z_matriz = []
    texto_celdas = []
    hover_celdas = []

    for e in etapas:
        icono = obtener_icono_cultivo(e["cultivo"])
        fila_z = []
        fila_txt = []
        fila_hov = []
        for idx_w, info_sem in enumerate(e["semanas_48"], start=1):
            mes_num = ((idx_w - 1) // 4) + 1
            sem_mes = ((idx_w - 1) % 4) + 1
            if not info_sem["activa"]:
                fila_z.append(0.0)
                fila_txt.append("·")
                fila_hov.append(f"Mes {mes_num} (S{sem_mes}) — Sin este cultivo")
            else:
                fase = info_sem["fase"]
                if fase == "Siembra":
                    fila_z.append(0.35)
                elif fase == "Crecimiento":
                    fila_z.append(0.70)
                else:
                    fila_z.append(1.0)
                fila_txt.append(icono)
                fila_hov.append(
                    f"<b>{icono} {e['cultivo']}</b><br>Mes {mes_num} • Semana S{sem_mes} (Sem. {idx_w}/48)<br>Etapa: <b>{fase}</b>"
                )
        z_matriz.append(fila_z)
        texto_celdas.append(fila_txt)
        hover_celdas.append(fila_hov)

    escala_colores = [
        [0.0, "#F5F3EF"],
        [0.15, "#F5F3EF"],
        [0.35, "#B3E5FC"],
        [0.70, "#C8E6C9"],
        [1.0, "#FFE082"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matriz,
            x=semanas_x,
            y=etiquetas_y,
            text=texto_celdas,
            texttemplate="%{text}",
            textfont={"size": 15},
            hoverinfo="text",
            hovertext=hover_celdas,
            colorscale=escala_colores,
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            xgap=2,
            ygap=6,
        )
    )

    # Líneas divisorias verticales cada 4 semanas (Mes 1 a Mes 12) y títulos de Mes arriba
    for m in range(1, 13):
        centro_mes = (m - 1) * 4 + 2.5
        fig.add_annotation(
            x=centro_mes,
            y=len(etapas) - 0.38,
            text=f"<b>MES {m}</b>",
            showarrow=False,
            font=dict(size=11, color="#0B3D91", family="Arial Black"),
            yref="y",
        )
        if m < 12:
            fig.add_vline(
                x=m * 4 + 0.5,
                line_width=1.8,
                line_dash="solid",
                line_color="#8D6E63",
            )

    tick_vals = list(range(1, 49))
    tick_texts = [f"S{((w - 1) % 4) + 1}" for w in tick_vals]

    fig.update_layout(
        height=255,
        margin=dict(l=10, r=10, t=38, b=28),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickfont=dict(size=9, color="#1C2819"),
            title=dict(
                text="<b>💧 Azul = Siembra/Trasplante   |   🌱 Verde = Crecimiento y Cobertura   |   🌾 Dorado = Cosecha   (48 Semanas = 0% Suelo Desnudo)</b>",
                font=dict(size=12, color="#1B5E20"),
            ),
            range=[0.5, 48.5],
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1C2819"),
        ),
    )
    return fig


def construir_grafico_clima_suelo(df_hist: pd.DataFrame, df_proj: pd.DataFrame, zona: str) -> go.Figure:
    """Gráfico histórico + proyección a 12 meses con colores de alto contraste (Azul Agua, Café Tierra, Verde Hoja)."""
    df_h = df_hist.tail(365).copy()
    df_p = df_proj.copy()

    df_h["hum_raiz_suav"] = df_h["humedad_suelo_raiz"].rolling(7, min_periods=1).mean()
    df_h["hum_sup_suav"] = df_h["humedad_suelo_sup"].rolling(7, min_periods=1).mean()
    df_h["temp_media_suav"] = df_h["temp_media_c"].rolling(7, min_periods=1).mean()
    df_h["temp_suelo_suav"] = (
        df_h["temp_suelo_c"].rolling(7, min_periods=1).mean()
        if "temp_suelo_c" in df_h.columns
        else df_h["temp_media_suav"]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"💧 Agua en las Raíces y Vigor Vegetal ({zona}) — Histórico + Proyección 12 Meses (48 Semanas)",
            f"🌡️ Temperatura de la Tierra y del Aire ({zona}) — Proyección 12 Meses",
        ),
        row_heights=[0.52, 0.48],
    )

    # Panel 1: Agua en el suelo + NDVI Sentinel-2
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["hum_raiz_suav"] * 100,
            mode="lines",
            name="💧 Humedad Raíces (NASA GWETROOT)",
            line=dict(color="#0277BD", width=3.2),
            fill="tozeroy",
            fillcolor="rgba(2, 119, 189, 0.14)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=df_p["humedad_suelo_raiz"] * 100,
            mode="lines",
            name="🌱 Humedad Raíces Proyectada (12 Meses)",
            line=dict(color="#2E7D32", width=3.5, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(46, 125, 50, 0.14)",
        ),
        row=1,
        col=1,
    )
    if "NDVI" in df_h.columns and "NDVI" in df_p.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df_h["fecha"], df_p["fecha"]]),
                y=pd.concat([df_h["NDVI"].rolling(7, min_periods=1).mean(), df_p["NDVI"]]) * 100,
                mode="lines",
                name="🛰️ Vigor Vegetal Sentinel-2 (NDVI x100)",
                line=dict(color="#558B2F", width=2.0, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Tierra y Aire
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["temp_suelo_suav"],
            mode="lines",
            name="🤎 Temp. de la Tierra (TS °C)",
            line=dict(color="#6D4C41", width=2.8),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["temp_media_suav"],
            mode="lines",
            name="☀️ Temp. del Aire (T2M °C)",
            line=dict(color="#F57C00", width=2.0, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=df_p["temp_suelo_c"] if "temp_suelo_c" in df_p.columns else df_p["temp_media_c"],
            mode="lines",
            name="🚀 Proyección Temp. Tierra (12 Meses)",
            line=dict(color="#D84315", width=3.2, dash="dash"),
        ),
        row=2,
        col=1,
    )

    fecha_corte = df_h["fecha"].max()
    fig.add_vline(
        x=fecha_corte,
        line_width=2.2,
        line_dash="dashdot",
        line_color="#0B3D91",
    )

    fig.update_yaxes(
        title_text="Agua en Suelo (%)",
        range=[15, 100],
        gridcolor="#E0E0E0",
        tickfont=dict(color="#1C2819"),
        title_font=dict(color="#01579B", size=13),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Grados (°C)",
        gridcolor="#E0E0E0",
        tickfont=dict(color="#1C2819"),
        title_font=dict(color="#4E342E", size=13),
        row=2,
        col=1,
    )
    fig.update_xaxes(gridcolor="#E0E0E0", tickfont=dict(color="#1C2819"))
    fig.update_layout(
        height=510,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
        font=dict(color="#1C2819"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            font=dict(color="#1C2819", size=12),
        ),
        hovermode="x unified",
    )
    return fig


def construir_mapa_interactivo_osm(
    lat_sel: float,
    lon_sel: float,
    ficha_predio: dict,
    cultivo_actual: str,
    primer_recomendado: str = None,
    poligono_guardado: list = None,
    zoom_inicial: int = 11,
) -> folium.Map:
    """Construye el mapa interactivo del Valle Central de Chile con OpenStreetMap +
    capa satelital Esri World Imagery, herramientas de dibujo de parcela (Draw),
    marcadores de las 4 estaciones base calibradas y resaltado exacto del polígono o punto elegido.
    """
    m = folium.Map(
        location=[lat_sel, lon_sel],
        zoom_start=zoom_inicial,
        tiles=None,
        control_scale=True,
    )

    # Capas base: OpenStreetMap + Satélite Esri World Imagery (ver potreros reales)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Mapa de Caminos y Comunas (OSM)",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Vista Satelital de Potreros (Esri)",
        control=True,
    ).add_to(m)

    # Herramienta para que el agricultor pueda dibujar su potrero (polígono/rectángulo) o marcar punto
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "polygon": {
                "allowIntersection": False,
                "shapeOptions": {"color": "#2E7D32", "fillColor": "#66BB6A", "fillOpacity": 0.40},
            },
            "rectangle": {
                "shapeOptions": {"color": "#0288D1", "fillColor": "#29B6F6", "fillOpacity": 0.40},
            },
            "marker": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Marcadores discretos de nuestras 4 estaciones base calibradas (Quillota, Rancagua, Talca, Chillán)
    for nombre_z, info_z in ZONAS_CHILE.items():
        folium.CircleMarker(
            location=[info_z["lat"], info_z["lon"]],
            radius=6,
            color="#0B3D91",
            weight=2,
            fill=True,
            fill_color="#1E88E5",
            fill_opacity=0.85,
            tooltip=f"📡 Estación Base NASA: {nombre_z} ({info_z['region']})",
        ).add_to(m)

    icono_act = obtener_icono_cultivo(cultivo_actual)
    icono_rec = obtener_icono_cultivo(primer_recomendado) if primer_recomendado else "🌱"
    texto_rec = f"{icono_rec} <b>Rotar a:</b> {primer_recomendado}" if primer_recomendado else ""

    popup_html = (
        f"<div style='font-family:sans-serif; font-size:13px; min-width:210px; color:#1C2819;'>"
        f"<div style='background:#0B3D91; color:white; padding:5px 8px; border-radius:6px; font-weight:bold; margin-bottom:6px;'>"
        f"🛰️ Predio Evaluado</div>"
        f"<b>📍 Sector:</b> {ficha_predio['nombre_ubicacion']}<br>"
        f"<b>🧭 Coordenadas:</b> {lat_sel:.4f}°, {lon_sel:.4f}°<br>"
        f"<b>💧 Humedad Raíz (NASA):</b> {ficha_predio['etiqueta_humedad']}<br>"
        f"<b>🌿 Vigor Sentinel-2:</b> NDVI {ficha_predio['ndvi_actual']:.2f}<br>"
        f"<b>🌾 Cultivo Actual:</b> {icono_act} {cultivo_actual}<br>"
        f"<div style='margin-top:5px; padding:4px 6px; background:#E8F5E9; border-left:4px solid #2E7D32; border-radius:4px;'>"
        f"{texto_rec}</div>"
        f"</div>"
    )

    # Si el usuario dibujó un polígono/rectángulo, dibujar EXACTAMENTE su polígono (sin el círculo gigante)
    if poligono_guardado and len(poligono_guardado) >= 3:
        folium.Polygon(
            locations=poligono_guardado,
            color="#2E7D32",
            weight=3,
            fill=True,
            fill_color="#66BB6A",
            fill_opacity=0.38,
            tooltip=f"📐 Parcela Dibujada ({ficha_predio.get('area_ha', '')} ha) — Clic para ver ficha",
            popup=folium.Popup(popup_html, max_width=280, show=False),
        ).add_to(m)

    # Marcador limpio en el centroide/punto sin abrir popup gigante automáticamente (evita superposición)
    folium.Marker(
        location=[lat_sel, lon_sel],
        popup=folium.Popup(popup_html, max_width=280, show=False),
        tooltip=f"🌱 Predio Seleccionado ({lat_sel:.4f}°, {lon_sel:.4f}°) | Clic para abrir detalle",
        icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m


def main():
    inyectar_estilos_campo_claro()

    df_hortalizas = cargar_catalogo_hortalizas()
    lista_cultivos = df_hortalizas["cultivo"].tolist()

    # Inicializar estado de la parcela en session_state (por defecto Rancagua)
    if "parcela_lat" not in st.session_state:
        st.session_state["parcela_lat"] = -34.1708
        st.session_state["parcela_lon"] = -70.7444
        st.session_state["tipo_seleccion"] = "Lugar caracterizado: Rancagua"
        st.session_state["area_ha"] = None
        st.session_state["poligono_coords"] = None

    # Caracterizar automáticamente la ubicación actual (humedad, agua disponible, cobertura, clima)
    ficha_auto = caracterizar_parcela_automatica(
        lat=st.session_state["parcela_lat"],
        lon=st.session_state["parcela_lon"],
        tipo_seleccion=st.session_state["tipo_seleccion"],
        area_ha=st.session_state["area_ha"],
    )
    zona_sel = ficha_auto["zona_cercana"]
    agua_auto = ficha_auto["agua_inferida"]  # Determinada 100% automáticamente por NASA POWER + Sentinel-2

    # =========================================================================
    # BANNER SUPERIOR PRINCIPAL
    # =========================================================================
    st.markdown(
        f"""
        <div class="nasa-agro-banner">
            <div class="banner-left">
                <div style="font-size:3.2rem; line-height:1;">🛰️🌍🌾</div>
                <div>
                    <span class="nasa-pill">NASA SPACE APPS • TOUR GUIADO PASO A PASO</span>
                    <p class="banner-title">Asistente Satelital de Rotación de Cultivos — Valle Central de Chile</p>
                    <p class="banner-sub">
                        Sigue los <b>3 pasos</b> abajo para obtener tu Plan Anual de 12 Meses y diagnóstico satelital automático.
                    </p>
                </div>
            </div>
            <div style="text-align:right; background:rgba(255,255,255,0.18); padding:0.65rem 1rem; border-radius:12px;">
                <div style="font-size:0.76rem; color:#E3F2FD !important; font-weight:800;">SECTOR EVALUADO</div>
                <div style="font-size:1.1rem; font-weight:900; color:#FFFFFF !important;">📍 {zona_sel} ({ZONAS_CHILE[zona_sel]['region']})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # PASO 1 DEL TOUR: ELEGIR EL PUNTO O DIBUJAR PARCELA EN EL MAPA AMPLIO
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🗺️ PASO 1: Elige el Lugar de tu Campo o Dibuja el Área de tu Parcela</div>',
        unsafe_allow_html=True,
    )

    col_modo_ubi, col_selector_lugar = st.columns([1.15, 1.0])
    with col_modo_ubi:
        modo_ubicacion = st.radio(
            "¿Cómo prefieres indicar dónde está tu terreno?",
            options=[
                "📐 Dibujar el polígono/rectángulo de mi parcela en el mapa",
                "📍 Elegir uno de los lugares caracterizados o hacer clic rápido",
            ],
            horizontal=True,
            index=0,
        )
    es_modo_dibujo = "Dibujar" in modo_ubicacion

    with col_selector_lugar:
        lugar_predefinido = st.selectbox(
            "📌 O salta directamente a un lugar caracterizado del Valle Central:",
            options=["(Mantener ubicación actual en el mapa)"]
            + [f"{z} — Región de {ZONAS_CHILE[z]['region']}" for z in ZONAS_CHILE.keys()],
            index=0,
        )
        if lugar_predefinido != "(Mantener ubicación actual en el mapa)":
            nombre_z_elegida = lugar_predefinido.split(" — ")[0].strip()
            info_z_el = ZONAS_CHILE[nombre_z_elegida]
            if (
                abs(info_z_el["lat"] - st.session_state["parcela_lat"]) > 1e-4
                or abs(info_z_el["lon"] - st.session_state["parcela_lon"]) > 1e-4
            ):
                st.session_state["parcela_lat"] = info_z_el["lat"]
                st.session_state["parcela_lon"] = info_z_el["lon"]
                st.session_state["tipo_seleccion"] = f"Lugar caracterizado: {nombre_z_elegida}"
                st.session_state["area_ha"] = None
                st.session_state["poligono_coords"] = None
                st.rerun()

    if es_modo_dibujo:
        st.caption(
            "💡 **Cómo dibujar sin interrupciones:** Usa el botón de **Polígono (pentágono)** o **Rectángulo (cuadrado)** arriba a la izquierda del mapa. "
            "Marca tus esquinas con calma: **el mapa NO se recargará mientras trazas los puntos**, solo cuando cierres la figura."
        )
        objetos_escuchados = ["last_active_drawing"]
    else:
        st.caption(
            "💡 **Modo Selección Rápida:** Haz un clic sobre tu potrero en el mapa o elige una comuna arriba."
        )
        objetos_escuchados = ["last_clicked", "last_active_drawing"]

    # Mapa de ANCHO COMPLETO (520px de alto) para que sea cómodo dibujar y sin textos superpuestos
    mapa_osm = construir_mapa_interactivo_osm(
        lat_sel=st.session_state["parcela_lat"],
        lon_sel=st.session_state["parcela_lon"],
        ficha_predio=ficha_auto,
        cultivo_actual="Hortaliza",
        primer_recomendado=None,
        poligono_guardado=st.session_state["poligono_coords"],
        zoom_inicial=12 if st.session_state["poligono_coords"] else 9,
    )
    salida_mapa = st_folium(
        mapa_osm,
        height=520,
        use_container_width=True,
        key="mapa_tour_paso1",
        returned_objects=objetos_escuchados,
    )

    nueva_coord = extraer_coordenadas_mapa(
        salida_mapa, permitir_clic_simple=(not es_modo_dibujo)
    )
    if nueva_coord is not None:
        n_lat, n_lon, n_tipo, n_area, n_poligono = nueva_coord
        cambio_posicion = (
            abs(n_lat - st.session_state["parcela_lat"]) > 1e-4
            or abs(n_lon - st.session_state["parcela_lon"]) > 1e-4
        )
        cambio_poligono = (n_poligono is not None) and (
            n_poligono != st.session_state["poligono_coords"]
        )
        if cambio_posicion or cambio_poligono:
            st.session_state["parcela_lat"] = n_lat
            st.session_state["parcela_lon"] = n_lon
            st.session_state["tipo_seleccion"] = n_tipo
            st.session_state["area_ha"] = n_area
            st.session_state["poligono_coords"] = n_poligono
            st.rerun()

    # Barra compacta de Diagnóstico Automático del Predio debajo del mapa (sin superponerse)
    area_txt = f"📐 Parcela de ~{ficha_auto['area_ha']} ha" if ficha_auto["area_ha"] else f"📍 {ficha_auto['tipo_seleccion']}"
    st.markdown(
        f"""
        <div class="tour-step-box" style="background:linear-gradient(90deg, #E8F5E9 0%, #E1F5FE 100%); border-color:#2E7D32;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                <div>
                    <span class="step-badge" style="background:#2E7D32;">🛰️ DIAGNÓSTICO AUTOMÁTICO DEL PUNTO</span>
                    <b style="font-size:1.08rem; color:#1C2819 !important;">{ficha_auto['nombre_ubicacion']}</b>
                    <span style="font-size:0.85rem; color:#4E342E !important; font-weight:700;"> ({ficha_auto['lat']:.4f}°, {ficha_auto['lon']:.4f}° • {area_txt})</span>
                </div>
                <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
                    <span style="background:#FFFFFF; border:1.5px solid #0288D1; color:#01579B !important; padding:4px 10px; border-radius:8px; font-size:0.85rem; font-weight:800;">
                        💧 Humedad Suelo Detectada: {ficha_auto['etiqueta_humedad']} (Agua {agua_auto})
                    </span>
                    <span style="background:#FFFFFF; border:1.5px solid #2E7D32; color:#1B5E20 !important; padding:4px 10px; border-radius:8px; font-size:0.85rem; font-weight:800;">
                        🌿 Vigor Sentinel-2: NDVI {ficha_auto['ndvi_actual']:.2f}
                    </span>
                    <span style="background:#FFFFFF; border:1.5px solid #6D4C41; color:#4E342E !important; padding:4px 10px; border-radius:8px; font-size:0.85rem; font-weight:800;">
                        🤎 Temp. Tierra: {ficha_auto['temp_suelo_c']:.1f} °C
                    </span>
                    <span style="background:#FFFFFF; border:1.5px solid #F57C00; color:#E65100 !important; padding:4px 10px; border-radius:8px; font-size:0.85rem; font-weight:800;">
                        ❄️ Heladas: {ficha_auto['riesgo_heladas_txt'].split()[0]}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # PASO 2 Y PASO 3 DEL TOUR: ELEGIR CULTIVO ACTUAL Y PRIORIDAD DE ROTACIÓN
    # =========================================================================
    col_paso2, col_paso3 = st.columns(2)

    with col_paso2:
        st.markdown(
            '<div class="section-header header-leaf">🌾 PASO 2: Elige tu Cultivo Actual (o Recién Cosechado)</div>',
            unsafe_allow_html=True,
        )
        cultivo_sel = st.selectbox(
            "¿Qué hortaliza tienes sembrada o acabas de sacar de este potrero?",
            options=lista_cultivos,
            format_func=lambda c: f"{obtener_icono_cultivo(c)} {c}",
            index=1 if "Tomate" in lista_cultivos else 0,
        )
        datos_c_act = df_hortalizas[df_hortalizas["cultivo"] == cultivo_sel].iloc[0]
        st.markdown(
            f"""
            <div class="tour-step-box" style="border-left:6px solid #2E7D32;">
                <div style="display:flex; align-items:center; gap:0.8rem;">
                    <span style="font-size:2.7rem;">{obtener_icono_cultivo(cultivo_sel)}</span>
                    <div>
                        <div style="font-size:1.15rem; font-weight:900; color:#1C2819 !important;">{cultivo_sel}</div>
                        <div style="font-size:0.84rem; font-weight:700; color:#5D4037 !important;">
                            🌿 Familia: <b>{datos_c_act['familia_botanica']}</b> • Efecto en suelo: <b>{datos_c_act['aporte_suelo']}</b>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_paso3:
        st.markdown(
            '<div class="section-header header-earth">🎯 PASO 3: Elige tu Prioridad de Rotación</div>',
            unsafe_allow_html=True,
        )
        opciones_prio = [
            "💧 Ahorrar agua",
            "🤎 Recuperar salud del suelo / nitrógeno",
            "☀️ Estabilidad térmica (evitar heladas)",
        ]
        idx_prio_sugerida = (
            opciones_prio.index(ficha_auto["prioridad_inferida"])
            if ficha_auto["prioridad_inferida"] in opciones_prio
            else 0
        )
        prioridad_sel = st.radio(
            f"¿Qué objetivo quieres priorizar para los próximos 12 meses? (Sugerencia satelital: {ficha_auto['prioridad_inferida']})",
            options=opciones_prio,
            index=idx_prio_sugerida,
        )
        st.markdown(
            f"""
            <div class="tour-step-box" style="border-left:6px solid #6D4C41; background:#EFEBE9;">
                <div style="font-size:0.86rem; font-weight:700; color:#3E2723 !important;">
                    🤖 <b>Disponibilidad hídrica calculada automáticamente por NASA POWER:</b> <b>{agua_auto}</b> ({ficha_auto['etiqueta_humedad']}).<br>
                    💡 <b>Criterio satelital:</b> {ficha_auto['motivo_prioridad']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ejecutar motor con los 3 pasos (Lugar + Cultivo Actual + Prioridad) y el agua auto-detectada
    resultado = generar_escenarios_rotacion(
        zona=zona_sel,
        nombre_cultivo_actual=cultivo_sel,
        disponibilidad_agua=agua_auto,
        prioridad_usuario=prioridad_sel,
        df_hist_personalizado=ficha_auto["df_interpolado"],
    )
    ind = resultado["indicadores"]
    cultivo_act = resultado["cultivo_actual"]
    ganador = resultado["recomendacion_principal"]
    icono_actual = obtener_icono_cultivo(cultivo_sel)
    icono_ganador = obtener_icono_cultivo(ganador["cultivo_recomendado"])

    # =========================================================================
    # RESULTADO PRINCIPAL: PLAN DE ROTACIÓN RECOMENDADO + 1 ÚNICA CARTA GANTT SIN DESLIZAR
    # =========================================================================
    sec_12m = resultado["secuencia_optima_12m"]
    etapas_12m = sec_12m["etapas"]

    st.markdown(
        '<div class="section-header header-leaf">🏆 RESULTADO: Tu Plan Maestro de Rotación Recomendado (Próximos 12 Meses)</div>',
        unsafe_allow_html=True,
    )

    e1, e2, e3 = etapas_12m[0], etapas_12m[1], etapas_12m[2]
    ic1, ic2, ic3 = (
        obtener_icono_cultivo(e1["cultivo"]),
        obtener_icono_cultivo(e2["cultivo"]),
        obtener_icono_cultivo(e3["cultivo"]),
    )

    st.markdown(
        f"""
        <div class="sequence-master-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem; margin-bottom:0.8rem;">
                <div>
                    <span style="background:#2E7D32; color:white !important; padding:0.3rem 0.85rem; border-radius:999px; font-size:0.8rem; font-weight:900;">
                        🏆 SECUENCIA ANUAL CONSOLIDADA (NASA POWER + COPERNICUS SENTINEL-2)
                    </span>
                    <h3 style="margin:0.4rem 0 0 0; font-size:1.5rem; font-weight:900; color:#1C2819 !important;">
                        {icono_actual} {cultivo_sel} (Actual) ➔ {ic1} {e1['cultivo']} ➔ {ic2} {e2['cultivo']} ➔ {ic3} {e3['cultivo']}
                    </h3>
                </div>
                <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
                    <div class="v-chip chip-water" style="min-width:135px; text-align:center;">
                        <span class="label">Ahorro Agua Anual</span>
                        <span class="val">💧 +{sec_12m['ahorro_hidrico_anual_pct']:.0f}% (Kc {sec_12m['kc_medio_anual']})</span>
                    </div>
                    <div class="v-chip chip-earth" style="min-width:145px; text-align:center;">
                        <span class="label">Riesgo Suelo Desnudo</span>
                        <span class="val">🤎 Baja de {sec_12m['suelo_desnudo_actual_pct']:.0f}% a 0%</span>
                    </div>
                    <div class="v-chip chip-leaf" style="min-width:135px; text-align:center;">
                        <span class="label">Cobertura Anual</span>
                        <span class="val">🌱 48 / 48 Semanas</span>
                    </div>
                </div>
            </div>
            <p style="font-size:0.95rem; font-weight:600; color:#263238 !important; margin:0 0 0.85rem 0; background:rgba(255,255,255,0.85); padding:0.65rem 0.9rem; border-radius:10px; border-left:5px solid #0288D1;">
                💡 <b>¿Por qué funciona esta secuencia?</b> {sec_12m['resumen_explicativo']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Bloques encadenados de la Secuencia 12 Meses
    cols_seq = st.columns(3)
    colores_borde_seq = ["#0288D1", "#2E7D32", "#6D4C41"]
    for idx_s, etapa in enumerate(etapas_12m):
        ic_s = obtener_icono_cultivo(etapa["cultivo"])
        c_borde = colores_borde_seq[idx_s]
        with cols_seq[idx_s]:
            st.markdown(
                f"""
                <div class="seq-step-box" style="border-color:{c_borde}; border-left: 6px solid {c_borde};">
                    <div style="font-size:0.78rem; font-weight:900; color:{c_borde} !important; text-transform:uppercase;">
                        ETAPA {etapa['orden']} • {etapa['etiqueta_periodo']} (Semanas {etapa['semana_inicio']} a {etapa['semana_fin']})
                    </div>
                    <div style="display:flex; align-items:center; gap:0.6rem; margin:0.35rem 0;">
                        <span style="font-size:2.4rem;">{ic_s}</span>
                        <div>
                            <div style="font-size:1.25rem; font-weight:900; color:#1C2819 !important;">{etapa['cultivo']}</div>
                            <div style="font-size:0.82rem; font-weight:700; color:#5D4037 !important;">🌿 {etapa['familia_botanica']}</div>
                        </div>
                    </div>
                    <div style="font-size:0.83rem; font-weight:700; color:#1C2819 !important; margin-top:0.3rem;">
                        💧 <b>Agua:</b> {etapa['demanda_hidrica']} (Kc={etapa['kc_promedio']}) &nbsp;|&nbsp;
                        🤎 <b>Suelo:</b> {etapa['aporte_suelo']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- ÚNICA CARTA GANTT VISUAL QUE MUESTRA EL AÑO COMPLETO SIN DESLIZAR ---
    st.markdown(
        '<div class="section-header header-earth">🗓️ Carta Gantt Anual (12 Meses / 48 Semanas a Vista Completa)</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(construir_figura_gantt_plotly(sec_12m), use_container_width=True)

    # Conservamos intactas las 3 tarjetas comparativas de escenarios en un bloque desplegable limpio
    with st.expander("🔍 Ver Comparativa Detallada frente al Monocultivo y Otras Alternativas Individuales", expanded=False):
        cols_esc = st.columns(3)
        estilos_tarjeta = ["crop-card-danger", "crop-card-water", "crop-card-winner"]
        iconos_medalla = ["⛔ NO REPETIR", "💧 OPCIÓN AHORRO AGUA", "🌱 OPCIÓN NUTRE LA TIERRA"]

        for idx, esc in enumerate(resultado["escenarios"]):
            es_ganador = esc["titulo"] == ganador["titulo"]
            clase_card = "crop-card-winner" if es_ganador else estilos_tarjeta[idx]
            icono_c = obtener_icono_cultivo(esc["cultivo_recomendado"])
            det = esc["detalles"]
            viab = int(round(esc["puntaje_viabilidad"]))

            if esc["demanda_hidrica"] == "Baja":
                gotas_txt = "💧 Baja (Poca Agua)"
            elif esc["demanda_hidrica"] == "Media":
                gotas_txt = "💧💧 Media"
            else:
                gotas_txt = "💧💧💧 Alta (Mucha Agua)"

            color_barra = "#E53935" if idx == 0 else ("#0288D1" if idx == 1 else "#2E7D32")
            etiqueta_top = "🏆 MEJOR INICIO DE CICLO" if es_ganador else iconos_medalla[idx]

            if idx == 0:
                consejo_corto = f"⚠️ Cuidado: Si repites {cultivo_sel}, las plagas quedan en la tierra y rinde menos."
            elif idx == 1:
                consejo_corto = f"💧 Ideal con poca agua: gasta menos riego y limpia las plagas de {cultivo_sel}."
            else:
                consejo_corto = f"🤎 Abono natural: deja la tierra descansada y con alimento (Nitrógeno) para después."

            with cols_esc[idx]:
                st.markdown(
                    f"""
                    <div class="crop-card {clase_card}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="background:{color_barra}; color:white !important; padding:0.25rem 0.75rem; border-radius:999px; font-size:0.78rem; font-weight:900;">
                                {etiqueta_top}
                            </span>
                            <span style="font-size:1.1rem; font-weight:900; color:{color_barra} !important;">
                                {viab}% Éxito
                            </span>
                        </div>
                        <div style="display:flex; align-items:center; margin:0.4rem 0;">
                            <span class="crop-emoji-big">{icono_c}</span>
                            <div>
                                <p class="crop-title-big">{esc['cultivo_recomendado']}</p>
                                <span style="font-size:0.85rem; font-weight:700; color:#5D4037 !important;">
                                    🌿 Familia: {esc['familia_botanica']}
                                </span>
                            </div>
                        </div>
                        <div class="progress-track">
                            <div style="width:{viab}%; background:{color_barra}; height:100%; border-radius:999px;"></div>
                        </div>
                        <div class="visual-chips-grid">
                            <div class="v-chip chip-water">
                                <span class="label">Consumo de Agua</span>
                                <span class="val">{gotas_txt}</span>
                            </div>
                            <div class="v-chip chip-earth">
                                <span class="label">Aporte a la Tierra</span>
                                <span class="val">🤎 {esc['aporte_suelo'].split()[0]}</span>
                            </div>
                            <div class="v-chip chip-leaf">
                                <span class="label">Tiempo a Cosecha</span>
                                <span class="val">🗓️ {esc['ciclo_dias']} días</span>
                            </div>
                            <div class="v-chip chip-sun">
                                <span class="label">Riesgo Heladas</span>
                                <span class="val">❄️ {det['dias_riesgo_helada']} días</span>
                            </div>
                        </div>
                        <div class="short-tip" style="border-left-color:{color_barra};">
                            {consejo_corto}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # SECCIÓN 2: UNA FIGURA PARA CADA VARIABLE SATELITAL (AGUA, TIERRA, SOL)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🛰️ Monitoreo Satelital NASA POWER y Copernicus Sentinel-2 (1 Figura por Variable)</div>',
        unsafe_allow_html=True,
    )

    # Fila 1 de medidores: 4 variables principales (Agua Raíz, Temp Tierra, Lluvia, Calor Solar GDD)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Agua en Raíces",
                icono="💧",
                valor=round(ind["humedad_raiz_actual"] * 100, 1),
                sufijo="%",
                min_val=0,
                max_val=100,
                color_barra="#0288D1",
                color_fondo_rango="#E1F5FE",
                subtitulo=f"NASA GWETROOT (Proy 6m: {ind['humedad_raiz_proj']*100:.0f}%)",
            ),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Temp. de la Tierra",
                icono="🤎",
                valor=ind["temp_suelo_actual_c"],
                sufijo="°C",
                min_val=0,
                max_val=35,
                color_barra="#6D4C41",
                color_fondo_rango="#EFEBE9",
                subtitulo=f"NASA TS (Proy 6m: {ind['temp_suelo_proj_c']:.1f} °C)",
            ),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Calor Acumulado",
                icono="☀️",
                valor=ind["gdd_proj_120"],
                sufijo="GDD",
                min_val=0,
                max_val=1800,
                color_barra="#2E7D32",
                color_fondo_rango="#E8F5E9",
                subtitulo="Energía para Crecer (Próx. 120 días)",
            ),
            use_container_width=True,
        )
    with g4:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Evaporación Diaria",
                icono="💨",
                valor=ind["evap_diaria_actual_mm"],
                sufijo="mm/d",
                min_val=0,
                max_val=6,
                color_barra="#F57C00",
                color_fondo_rango="#FFF8E1",
                subtitulo=f"NASA EVLAND ({ind['evap_acum_120d_mm']:.0f} mm en 120d)",
            ),
            use_container_width=True,
        )

    # Fila 2 de medidores: Las demás variables del satélite NASA (Radiación, Humedad Aire, Lluvia, Frío)
    g5, g6, g7, g8 = st.columns(4)
    with g5:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Luz del Sol",
                icono="🔆",
                valor=ind["radiacion_actual"],
                sufijo="MJ",
                min_val=0,
                max_val=35,
                color_barra="#FBC02D",
                color_fondo_rango="#FFFDE7",
                subtitulo="Radiación Solar (ALLSKY_SFC_SW_DWN)",
            ),
            use_container_width=True,
        )
    with g6:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Humedad del Aire",
                icono="☁️",
                valor=ind["humedad_rel_actual_pct"],
                sufijo="%",
                min_val=0,
                max_val=100,
                color_barra="#039BE5",
                color_fondo_rango="#E1F5FE",
                subtitulo="NASA RH2M (Riesgo de Hongos >75%)",
            ),
            use_container_width=True,
        )
    with g7:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Lluvia del Año",
                icono="🌧️",
                valor=ind["precip_ultimo_anio_mm"],
                sufijo="mm",
                min_val=0,
                max_val=1200,
                color_barra="#1565C0",
                color_fondo_rango="#E3F2FD",
                subtitulo=f"Media normal: {ind['precip_media_anual_mm']:.0f} mm/año",
            ),
            use_container_width=True,
        )
    with g8:
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Días de Helada",
                icono="❄️",
                valor=ind["dias_riesgo_cultivo_proj"],
                sufijo="días",
                min_val=0,
                max_val=30,
                color_barra="#00ACC1",
                color_fondo_rango="#E0F7FA",
                subtitulo="Riesgo de Frío en Próximos 12 Meses",
            ),
            use_container_width=True,
        )

    # --- TARJETAS DISCRETAS COPERNICUS SENTINEL-2 (ESA) ---
    st.markdown(
        '<div style="font-size:0.95rem; font-weight:900; color:#1B5E20; margin:0.35rem 0 0.45rem 0;">🛰️ Índices Ópticos de Vegetación y Suelo — Copernicus Sentinel-2 L2A (10–20m)</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
            <div class="sentinel-badge-card" style="border-left:6px solid #2E7D32;">
                <div style="font-size:0.75rem; font-weight:800; color:#2E7D32 !important;">🌿 VIGOR VEGETATIVO (NDVI)</div>
                <div style="font-size:1.55rem; font-weight:900; color:#1C2819 !important;">{ind['ndvi_actual']:.2f} <span style="font-size:0.82rem; font-weight:700; color:#2E7D32 !important;">(Proy 12m: {ind['ndvi_proj_12m']:.2f})</span></div>
                <div style="font-size:0.76rem; color:#4E342E !important;">Bandas NIR/Rojo (B08-B04) • Verdes intensos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="sentinel-badge-card" style="border-left:6px solid #558B2F;">
                <div style="font-size:0.75rem; font-weight:800; color:#558B2F !important;">🌱 BIOMASA MEJORADA (EVI)</div>
                <div style="font-size:1.55rem; font-weight:900; color:#1C2819 !important;">{ind['evi_actual']:.2f} <span style="font-size:0.82rem; font-weight:700; color:#558B2F !important;">Alta sensibilidad</span></div>
                <div style="font-size:0.76rem; color:#4E342E !important;">Corrige atmósfera y suelo en follaje denso</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="sentinel-badge-card" style="border-left:6px solid #0288D1;">
                <div style="font-size:0.75rem; font-weight:800; color:#0288D1 !important;">💧 AGUA EN LA HOJA (NDMI)</div>
                <div style="font-size:1.55rem; font-weight:900; color:#1C2819 !important;">{ind['ndmi_actual']:+.2f} <span style="font-size:0.82rem; font-weight:700; color:#0288D1 !important;">(Proy 12m: {ind['ndmi_proj_12m']:+.2f})</span></div>
                <div style="font-size:0.76rem; color:#4E342E !important;">Bandas NIR/SWIR (B08-B11) • Estrés foliar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""
            <div class="sentinel-badge-card" style="border-left:6px solid #6D4C41;">
                <div style="font-size:0.75rem; font-weight:800; color:#6D4C41 !important;">🤎 RIESGO SUELO DESNUDO</div>
                <div style="font-size:1.55rem; font-weight:900; color:#1C2819 !important;">{ind['suelo_desnudo_pct']:.0f}% <span style="font-size:0.82rem; font-weight:700; color:#2E7D32 !important;">➔ Meta: 0%</span></div>
                <div style="font-size:0.76rem; color:#4E342E !important;">{ind['suelo_desnudo_dias_anual']} días/año sin cobertura vegetal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # SECCIÓN 3: DIAGRAMAS VISUALES (CORTE DE LA TIERRA + ESTRELLA COMPARATIVA)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-earth">🤎 Salud de tu Tierra y Pronóstico Agroclimático para los Próximos 12 Meses</div>',
        unsafe_allow_html=True,
    )

    col_tierra, col_radar = st.columns([1.05, 0.95])
    with col_tierra:
        fig_perfil = crear_figura_perfil_tierra_agua(ind, zona_sel)
        st.plotly_chart(fig_perfil, use_container_width=True)

    with col_radar:
        fig_radar = crear_radar_comparativo_escenarios(resultado["escenarios"])
        st.plotly_chart(fig_radar, use_container_width=True)

    # Gráfico temporal de evolución y proyección 6 meses
    fig_evol = construir_grafico_clima_suelo(
        resultado["df_historico"], resultado["df_proyeccion"], zona_sel
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    # =========================================================================
    # SECCIÓN 4: PANEL PARA USUARIOS CON EXPERIENCIA / AGRÓNOMOS (COMPLEMENTARIO)
    # =========================================================================
    with st.expander("🔬 ¿Eres agricultor con experiencia o técnico agrícola? Haz clic aquí para ver las tablas técnicas completas"):
        t1, t2 = st.tabs([
            "📊 Tabla Técnica de Cultivos (Kc, GDD, Heladas)",
            "🛰️ Datos Crudos del Satélite NASA POWER + Sentinel-2",
        ])
        with t1:
            st.dataframe(df_hortalizas, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(
                resultado["df_historico"].tail(90).sort_values("fecha", ascending=False),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
