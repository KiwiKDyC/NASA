import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from descargar_datos import ZONAS_CHILE
from modelo_rotacion import (
    cargar_catalogo_hortalizas,
    generar_escenarios_rotacion,
)

st.set_page_config(
    page_title="AgroSatélite NASA | Rotación de Cultivos Chile",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
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
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EFECE4 0%, #E5DEC9 100%) !important;
            border-right: 3px solid #8D6E63 !important;
        }
        [data-testid="stSidebar"] * {
            color: #1C2819 !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
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


def construir_grafico_clima_suelo(df_hist: pd.DataFrame, df_proj: pd.DataFrame, zona: str) -> go.Figure:
    """Gráfico histórico + proyección a 6 meses con colores de alto contraste (Azul Agua, Café Tierra, Verde Hoja)."""
    df_h = df_hist.tail(300).copy()
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
            f"💧 Agua en las Raíces ({zona}) — Histórico + Proyección Satelital 6 Meses",
            f"🌡️ Temperatura de la Tierra y del Aire ({zona}) — Próximos 6 Meses",
        ),
        row_heights=[0.52, 0.48],
    )

    # Panel 1: Agua en el suelo
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["hum_raiz_suav"] * 100,
            mode="lines",
            name="💧 Humedad Raíces (Observado NASA)",
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
            name="🌱 Humedad Raíces Proyectada (6 Meses)",
            line=dict(color="#2E7D32", width=3.5, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(46, 125, 50, 0.14)",
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
            name="🚀 Proyección Temp. Tierra (6 Meses)",
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


def main():
    inyectar_estilos_campo_claro()

    df_hortalizas = cargar_catalogo_hortalizas()
    lista_cultivos = df_hortalizas["cultivo"].tolist()

    # =========================================================================
    # BARRA LATERAL: CONTROLES SIMPLES Y VISUALES PARA EL AGRICULTOR
    # =========================================================================
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; background:#0B3D91; color:white !important; padding:0.9rem; border-radius:14px; margin-bottom:1rem;">
                <div style="font-size:2.3rem;">🛰️🌱</div>
                <div style="font-weight:900; font-size:1.15rem; color:white !important;">MI POTRERO NASA</div>
                <div style="font-size:0.8rem; color:#BBDEFB !important;">Configura tu campo en 4 pasos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        zona_sel = st.selectbox(
            "1️⃣ ¿Dónde está tu campo?",
            options=list(ZONAS_CHILE.keys()),
            format_func=lambda z: f"📍 {z} ({ZONAS_CHILE[z]['region']})",
            index=0,
        )

        cultivo_sel = st.selectbox(
            "2️⃣ ¿Qué acabas de cosechar?",
            options=lista_cultivos,
            format_func=lambda c: f"{obtener_icono_cultivo(c)} {c}",
            index=1 if "Tomate" in lista_cultivos else 0,
        )

        agua_sel = st.select_slider(
            "3️⃣ ¿Cuánta agua tienes para regar?",
            options=["Baja", "Media", "Alta"],
            value="Baja",
        )

        prioridad_sel = st.radio(
            "4️⃣ ¿Qué es lo más importante ahora?",
            options=[
                "💧 Ahorrar agua",
                "🤎 Recuperar salud del suelo / nitrógeno",
                "☀️ Estabilidad térmica (evitar heladas)",
            ],
            index=0,
        )

        st.divider()
        modo_vista = st.toggle(
            "🔬 Activar Modo Experto / Agrónomo",
            value=False,
            help="Muestra tablas técnicas, coeficientes Kc, GDD detallados y datos crudos del satélite NASA.",
        )

        info_z = ZONAS_CHILE[zona_sel]
        st.markdown(
            f"""
            <div style="background:#FFFFFF; padding:0.75rem; border-radius:12px; border:1.5px solid #BCAAA4; font-size:0.84rem;">
                <b>📡 Satélite NASA POWER Activo</b><br>
                📍 <b>Comuna:</b> {zona_sel}<br>
                🧭 <b>Coord:</b> {info_z['lat']}°, {info_z['lon']}°<br>
                📶 <b>Estado:</b> 100% Offline Listo
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ejecutar motor
    resultado = generar_escenarios_rotacion(
        zona=zona_sel,
        nombre_cultivo_actual=cultivo_sel,
        disponibilidad_agua=agua_sel,
        prioridad_usuario=prioridad_sel,
    )
    ind = resultado["indicadores"]
    cultivo_act = resultado["cultivo_actual"]
    ganador = resultado["recomendacion_principal"]
    icono_actual = obtener_icono_cultivo(cultivo_sel)
    icono_ganador = obtener_icono_cultivo(ganador["cultivo_recomendado"])

    # =========================================================================
    # BANNER SUPERIOR CON ILUSTRACIÓN SVG SATÉLITE NASA -> CAMPO CHILENO
    # =========================================================================
    st.markdown(
        f"""
        <div class="nasa-agro-banner">
            <div class="banner-left">
                <div style="font-size:3.2rem; line-height:1;">🛰️🌍🌾</div>
                <div>
                    <span class="nasa-pill">NASA SPACE APPS • AGROCLIMATOLOGÍA SATELITAL</span>
                    <p class="banner-title">Guía Visual de Rotación de Cultivos — {zona_sel}</p>
                    <p class="banner-sub">
                        Vienes de <b>{icono_actual} {cultivo_sel}</b> ➔ El satélite recomienda sembrar:
                        <b style="background:#2E7D32; padding:2px 10px; border-radius:8px; color:#FFF !important;">
                            {icono_ganador} {ganador['cultivo_recomendado']}
                        </b>
                    </p>
                </div>
            </div>
            <div style="text-align:right; background:rgba(255,255,255,0.15); padding:0.65rem 1rem; border-radius:12px;">
                <div style="font-size:0.78rem; color:#E3F2FD !important; font-weight:700;">VIABILIDAD DEL CAMPO</div>
                <div style="font-size:1.9rem; font-weight:900; color:#FFFFFF !important;">{ganador['puntaje_viabilidad']:.0f}% ⭐</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECCIÓN 1: TARJETAS DE PRESENTACIÓN (¿QUÉ SIEMBRO AHORA?) - POCA LETRA
    # =========================================================================
    st.markdown(
        '<div class="section-header header-leaf">🌱 PASO 1: Elige tu Próximo Cultivo (Tarjetas de Decisión Rápida)</div>',
        unsafe_allow_html=True,
    )

    cols_esc = st.columns(3)
    estilos_tarjeta = ["crop-card-danger", "crop-card-water", "crop-card-winner"]
    iconos_medalla = ["⛔ NO REPETIR", "💧 OPCIÓN AHORRO AGUA", "🌱 OPCIÓN NUTRE LA TIERRA"]

    for idx, esc in enumerate(resultado["escenarios"]):
        es_ganador = esc["titulo"] == ganador["titulo"]
        clase_card = "crop-card-winner" if es_ganador else estilos_tarjeta[idx]
        icono_c = obtener_icono_cultivo(esc["cultivo_recomendado"])
        det = esc["detalles"]
        viab = int(round(esc["puntaje_viabilidad"]))

        # Símbolos intuitivos de gotas de agua
        if esc["demanda_hidrica"] == "Baja":
            gotas_txt = "💧 Baja (Poca Agua)"
        elif esc["demanda_hidrica"] == "Media":
            gotas_txt = "💧💧 Media"
        else:
            gotas_txt = "💧💧💧 Alta (Mucha Agua)"

        color_barra = "#E53935" if idx == 0 else ("#0288D1" if idx == 1 else "#2E7D32")
        etiqueta_top = "🏆 MEJOR OPCIÓN PARA TI" if es_ganador else iconos_medalla[idx]

        # Frase ultra-corta y directa para el pequeño agricultor
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
        '<div class="section-header header-water">🛰️ PASO 2: ¿Cómo ve el Satélite NASA tu Campo Hoy? (1 Figura por Variable)</div>',
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
                subtitulo="Riesgo de Frío en Próximos 6 Meses",
            ),
            use_container_width=True,
        )

    # =========================================================================
    # SECCIÓN 3: DIAGRAMAS VISUALES (CORTE DE LA TIERRA + ESTRELLA COMPARATIVA)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-earth">🤎 PASO 3: Salud de tu Tierra y Pronóstico para los Próximos 6 Meses</div>',
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
    if modo_vista:
        st.markdown(
            '<div class="section-header header-leaf">🔬 Panel Técnico para Agricultores Experimentados y Asesores</div>',
            unsafe_allow_html=True,
        )
        t1, t2 = st.tabs([
            "📊 Matriz Agronómica Completa (Kc, GDD, Familias)",
            "🛰️ Serie Diaria Satelital NASA POWER (11 Variables)",
        ])
        with t1:
            st.dataframe(df_hortalizas, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(
                resultado["df_historico"].tail(90).sort_values("fecha", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
    else:
        with st.expander("🔬 ¿Eres agricultor con experiencia o técnico agrícola? Haz clic aquí para ver los datos completos"):
            t1, t2 = st.tabs([
                "📊 Tabla Técnica de Cultivos (Kc, GDD, Heladas)",
                "🛰️ Datos Crudos del Satélite NASA POWER",
            ])
            with t1:
                st.dataframe(df_hortalizas, use_container_width=True, hide_index=True)
            with t2:
                st.dataframe(
                    resultado["df_historico"].tail(60).sort_values("fecha", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
