from typing import Tuple
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

# Iconografía visual para cada hortaliza y cultivo de cobertura
ICONOS_CULTIVOS = {
    "Papa": "🥔",
    "Tomate": "🍅",
    "Pimentón": "🫑",
    "Cebolla": "🧅",
    "Ajo": "🧄",
    "Lechuga": "🥬",
    "Brócoli / Coliflor": "🥦",
    "Repollo": "🥬",
    "Poroto verde": "🫛",
    "Haba / Arveja": "🌿",
    "Maíz dulce": "🌽",
    "Zapallo": "🎃",
    "Zanahoria": "🥕",
    "Vicia (Abono Verde)": "🌿",
    "Mostaza Blanca (Biofumigante)": "🌼",
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
        .stApp {
            background-color: #FBF9F5 !important;
            color: #1C2819 !important;
        }
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }

        .tour-step-box {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.1rem 1.35rem;
            border: 2.5px solid #C8E6C9;
            box-shadow: 0 4px 14px rgba(28, 40, 25, 0.06);
            margin-bottom: 1rem;
        }
        .step-badge {
            display: inline-block;
            background: #0B3D91;
            color: #FFFFFF !important;
            font-weight: 900;
            font-size: 0.84rem;
            padding: 0.28rem 0.85rem;
            border-radius: 999px;
            margin-right: 0.5rem;
        }

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

        .section-header {
            font-size: 1.28rem;
            font-weight: 900;
            padding: 0.6rem 1.1rem;
            border-radius: 12px;
            margin: 1.15rem 0 0.8rem 0;
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

        .visual-chips-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-top: 0.65rem;
            margin-bottom: 0.55rem;
        }
        .v-chip {
            border-radius: 12px;
            padding: 0.5rem 0.65rem;
            font-size: 0.86rem;
            font-weight: 700;
            display: flex;
            flex-direction: column;
        }
        .v-chip span.label {
            font-size: 0.71rem;
            font-weight: 800;
            text-transform: uppercase;
            opacity: 0.88;
        }
        .v-chip span.val {
            font-size: 0.98rem;
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
        .short-tip {
            font-size: 0.88rem;
            font-weight: 600;
            color: #263238 !important;
            background: rgba(255,255,255,0.92);
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            border-left: 4px solid #2E7D32;
            margin-top: 0.45rem;
            line-height: 1.38;
        }

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
            border-radius: 16px;
            padding: 1rem 1.1rem;
            border: 2.5px solid #A5D6A7;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            height: 100%;
        }
        .sentinel-badge-card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 0.9rem 1.05rem;
            border: 2px solid #B0BEC5;
            box-shadow: 0 3px 8px rgba(0,0,0,0.04);
            height: 100%;
        }
        .diag-item-card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            border: 2px solid #CFD8DC;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        /* Hacer más grandes los selectores para facilitar uso en terreno */
        div[data-baseweb="select"] > div {
            min-height: 48px !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
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
                "font": {"size": 25, "color": "#1C2819", "family": "Arial Black"},
            },
            title={
                "text": f"<b>{icono} {titulo}</b><br><span style='font-size:11px;color:#4E342E'>{subtitulo}</span>",
                "font": {"size": 14, "color": "#1C2819"},
            },
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#4E342E",
                    "tickfont": {"size": 10, "color": "#1C2819"},
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
        height=185,
        margin=dict(l=16, r=16, t=54, b=10),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1C2819"),
    )
    return fig


def traducir_valor_monitoreo(clave: str, valor: float, extra: float = 0.0) -> Tuple[str, str, str]:
    """Traduce en palabras sencillas qué significa cada valor del Monitoreo Satelital."""
    if clave == "agua_raices":
        if valor >= 64.0:
            return ("🟢 Nivel Alto", "Tierra bien húmeda en profundidad; excelente reserva para las raíces.", "#2E7D32")
        elif valor >= 52.0:
            return ("🟡 Nivel Medio", "Humedad suficiente hoy, pero conviene regar con eficiencia en verano.", "#0288D1")
        return ("🟠 Nivel Bajo", "Poca agua acumulada en raíces; prioriza cultivos ahorradores.", "#E65100")

    if clave == "temp_tierra":
        if valor >= 17.0:
            return ("🟢 Tierra Tibia", "Las semillas germinan rápido y las raíces absorben nutrientes con fuerza.", "#D84315")
        elif valor >= 12.0:
            return ("🟢 Tierra Templada", "Temperatura ideal para hortalizas de hoja, raíces, cebolla y leguminosas.", "#6D4C41")
        return ("🔵 Tierra Fría", "Las plantas crecen más lento; usa cultivos resistentes al invierno.", "#1565C0")

    if clave == "calor_gdd":
        if valor >= 1100:
            return ("☀️ Mucha Energía", "El sol acumula calor rápido; los cultivos se cosecharán en menos días.", "#2E7D32")
        elif valor >= 750:
            return ("🌤️ Energía Moderada", "Ritmo de crecimiento normal para la época del año.", "#F57C00")
        return ("🌥️ Energía Suave", "Acumulación térmica lenta; los ciclos demorarán algunas semanas más.", "#0277BD")

    if clave == "evaporacion":
        if valor >= 3.2:
            return ("🟠 Sed Alta", f"El sol evapora bastante agua ({extra:.1f} mm/d de referencia); protege el suelo con cobertura.", "#E65100")
        elif valor >= 1.8:
            return ("🟡 Sed Moderada", f"Pérdida de agua normal ({extra:.1f} mm/d de referencia); riego estándar.", "#F57C00")
        return ("🟢 Poca Pérdida", "El suelo pierde muy poca agua por evaporación directa.", "#2E7D32")

    if clave == "luz_solar":
        if valor >= 20.0:
            return ("🔆 Sol Intenso", "Máxima fotosíntesis para cultivos de fruto y hoja.", "#F9A825")
        elif valor >= 12.0:
            return ("🌤️ Buena Luz", "Radiación adecuada para el desarrollo parejo del cultivo.", "#558B2F")
        return ("☁️ Días Cortos", "Menor luz invernal; ideal para habas, ajo, cebolla o abonos verdes.", "#0288D1")

    if clave == "humedad_aire":
        if valor >= 76.0:
            return ("⚠️ Aire Húmedo", "Atención: sobre 75% de humedad en el aire aparecen hongos en hojas; evita repetir Tomate/Papa.", "#C62828")
        elif valor >= 55.0:
            return ("🟢 Aire Equilibrado", "Ambiente sano con bajo riesgo de enfermedades foliares.", "#2E7D32")
        return ("🟡 Aire Seco", "Bajo riesgo de hongos, pero las plantas transpiran más agua.", "#EF6C00")

    if clave == "lluvia":
        if valor >= 550:
            return ("🌧️ Buena Lluvia", "El agua caída recarga bien el perfil del suelo para el invierno.", "#1565C0")
        elif valor >= 350:
            return ("🌦️ Lluvia Media", "Aporte moderado de lluvias; apóyate en abonos verdes para infiltrar cada gota.", "#0288D1")
        return ("🌵 Año Seco", "Poca lluvia acumulada; es indispensable elegir cultivos de bajo consumo (Kc bajo).", "#E65100")

    if clave == "heladas":
        if valor == 0:
            return ("🟢 Sin Heladas", "No se proyectan días bajo el límite de daño de tu cultivo.", "#2E7D32")
        elif valor <= 5:
            return ("🟡 Heladas Leves", f"Se esperan ~{int(valor)} noches muy frías; protege almácigos sensibles.", "#00838F")
        return ("❄️ Frío Fuerte", f"Atención: ~{int(valor)} días con heladas fuertes; siembra especies resistentes al frío.", "#01579B")

    return ("🟢 Normal", "Valor dentro del rango esperado.", "#2E7D32")


def construir_leyenda_gantt_cuadrados_html() -> str:
    """Leyenda visual con cuadrados de color sólido según lo solicitado por el usuario."""
    return (
        "<div style='display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center; justify-content:center; "
        "background:#FFFFFF; border:2px solid #D7CCC8; border-radius:12px; padding:0.6rem 1.1rem; margin-bottom:0.5rem; font-size:0.92rem; font-weight:800; color:#1C2819;'>"
        "<span style='display:flex; align-items:center; gap:0.45rem;'>"
        "<span style='display:inline-block; width:18px; height:18px; background:#B3E5FC; border:2px solid #0288D1; border-radius:4px;'></span>"
        "<span style='color:#01579B;'>Azul = Siembra / Trasplante</span>"
        "</span>"
        "<span style='display:flex; align-items:center; gap:0.45rem;'>"
        "<span style='display:inline-block; width:18px; height:18px; background:#C8E6C9; border:2px solid #2E7D32; border-radius:4px;'></span>"
        "<span style='color:#1B5E20;'>Verde = Crecimiento y Desarrollo</span>"
        "</span>"
        "<span style='display:flex; align-items:center; gap:0.45rem;'>"
        "<span style='display:inline-block; width:18px; height:18px; background:#FFE082; border:2px solid #F57C00; border-radius:4px;'></span>"
        "<span style='color:#E65100;'>Dorado = Cosecha</span>"
        "</span>"
        "<span style='display:flex; align-items:center; gap:0.45rem;'>"
        "<span style='display:inline-block; width:18px; height:18px; background:rgba(242, 239, 233, 0.45); border:1.5px solid #E0DCD5; border-radius:4px;'></span>"
        "<span style='color:#78909C;'>Celda suave = Sin este cultivo en esa semana</span>"
        "</span>"
        "</div>"
    )


def construir_figura_gantt_plotly(
    datos_gantt: dict,
    modo_vista: str = "anio_1",
) -> go.Figure:
    """Construye la Carta Gantt interactiva en Plotly para cada año (12 meses / 48 semanas):
    - Eje Y: Solo número, ícono y nombre del cultivo (sin familia botánica ni aporte de Nitrógeno).
    - Celdas vacías: Color sólido suave/translúcido sin punto central para destacar las etapas activas.
    - Eje X superior: Semanas numeradas como '1, 2, 3, 4' debajo de 'MES 1, MES 2...' en la parte superior.
    - Ciclos dinámicos determinados por los Grados-Día (GDD) de la zona evaluada.
    """
    if modo_vista == "anio_1":
        etapas_base = datos_gantt["etapas_anio1"]
        offset_mes = 0
        offset_sem = 0
    else:
        etapas_base = datos_gantt["etapas_anio2"]
        offset_mes = 12
        offset_sem = 48

    etapas = list(reversed(etapas_base))
    # Solo mostrar número, ícono y nombre del cultivo (sin familia botánica ni aporte de nitrógeno)
    etiquetas_y = [
        f"<b>{e['orden']}. {obtener_icono_cultivo(e['cultivo'])} {e['cultivo']}</b>"
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
            mes_num = offset_mes + ((idx_w - 1) // 4) + 1
            sem_mes = ((idx_w - 1) % 4) + 1
            sem_global = offset_sem + idx_w
            if not info_sem["activa"]:
                fila_z.append(0.0)
                fila_txt.append("")  # Sin punto central para no distraer visualmente
                fila_hov.append(f"Mes {mes_num} • Semana {sem_mes} — Periodo inactivo para {e['cultivo']}")
            else:
                fase = info_sem["fase"]
                if fase == "Siembra":
                    fila_z.append(0.35)
                    nombre_fase = "Siembra / Trasplante"
                elif fase == "Crecimiento":
                    fila_z.append(0.70)
                    nombre_fase = "Crecimiento y Desarrollo"
                else:
                    fila_z.append(1.0)
                    nombre_fase = "Cosecha"
                fila_txt.append(icono)
                fila_hov.append(
                    f"<b>{icono} {e['cultivo']}</b><br>"
                    f"Mes {mes_num} • Semana {sem_mes} (Sem. {sem_global}/96)<br>"
                    f"Etapa: <b>{nombre_fase}</b><br>"
                    f"Ciclo ajustado por GDD zonal: <b>{e.get('ciclo_dinamico_dias', 90)} días ({e['duracion_semanas']} semanas)</b>"
                )
        z_matriz.append(fila_z)
        texto_celdas.append(fila_txt)
        hover_celdas.append(fila_hov)

    # Celdas vacías en tono crema muy pálido y translúcido; etapas activas en colores vivos
    escala_colores = [
        [0.0, "rgba(244, 241, 234, 0.38)"],
        [0.15, "rgba(244, 241, 234, 0.38)"],
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
            ygap=7,
        )
    )

    # Encabezados de MES 1..12 (o MES 13..24) arriba de los números de semana 1, 2, 3, 4
    for m in range(1, 13):
        centro_mes = (m - 1) * 4 + 2.5
        mes_real = offset_mes + m
        fig.add_annotation(
            x=centro_mes,
            y=1.16,
            xref="x",
            yref="paper",
            text=f"<b>MES {mes_real}</b>",
            showarrow=False,
            font=dict(size=11, color="#0B3D91" if offset_mes == 0 else "#1B5E20", family="Arial Black"),
        )
        if m < 12:
            fig.add_vline(
                x=m * 4 + 0.5,
                line_width=1.6,
                line_dash="solid",
                line_color="#BCAAA4",
            )

    tick_vals = list(range(1, 49))
    # Semanas numeradas solo como "1, 2, 3, 4" (quitando la "S") en la parte superior
    tick_texts = [f"{((w - 1) % 4) + 1}" for w in tick_vals]

    fig.update_layout(
        height=255,
        margin=dict(l=10, r=10, t=58, b=14),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            side="top",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickfont=dict(size=10, color="#3E2723", family="Arial"),
            range=[0.5, 48.5],
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#1C2819"),
        ),
    )
    return fig


def crear_figuras_cuenta_corriente_n_y_agua(cuenta: dict, cultivo_actual: str) -> Tuple[go.Figure, go.Figure]:
    """Crea los 2 gráficos mejorados de la Cuenta Corriente:
    1. Gráfico de Nitrógeno: Barras con el nombre del cultivo en las columnas, eje X con la
       distribución temporal (Meses/Año), y línea superpuesta que muestra la acumulación de Nitrógeno
       a lo largo del tiempo (reemplazando la tabla cartola).
    2. Gráfico de Agua Total en 2 Años (ET_c = Kc * ET_ref): Muestra directamente en el gráfico el
       dato de Reducción de Consumo Hídrico (%) y m³/ha ahorrados.
    """
    movs = cuenta["movimientos"]
    eje_x_temporal = [m["distribucion_temporal"] for m in movs]
    flujos_n = [m["flujo_n_kg_ha"] for m in movs]
    saldos_acum = [m["saldo_n_acum_kg_ha"] for m in movs]
    colores_n = ["#2E7D32" if v >= 0 else "#C62828" for v in flujos_n]
    textos_barras = [
        f"<b>{obtener_icono_cultivo(m['cultivo'])} {m['cultivo']}</b><br>{m['flujo_n_kg_ha']:+} kg N/ha"
        for m in movs
    ]

    fig_n = go.Figure()
    fig_n.add_trace(
        go.Bar(
            x=eje_x_temporal,
            y=flujos_n,
            marker=dict(color=colores_n, line=dict(color="#1C2819", width=1.4), opacity=0.88),
            text=textos_barras,
            textposition="outside",
            textfont=dict(size=11, color="#1C2819"),
            name="Aporte / Extracción por Cultivo (kg N/ha)",
            hovertemplate="<b>%{text}</b><br>Periodo: %{x}<extra></extra>",
        )
    )
    fig_n.add_trace(
        go.Scatter(
            x=eje_x_temporal,
            y=saldos_acum,
            mode="lines+markers+text",
            name="📈 Saldo Acumulado de Nitrógeno en el Suelo (kg N/ha)",
            line=dict(color="#0B3D91", width=3.8),
            marker=dict(size=10, color="#FFC107", line=dict(color="#0B3D91", width=2.5)),
            text=[f"<b>Acum: {s:+} kg N</b>" for s in saldos_acum],
            textposition="top center",
            textfont=dict(size=11, color="#0B3D91"),
            hovertemplate="Periodo: %{x}<br><b>Saldo Acumulado: %{y:+} kg N/ha</b><extra></extra>",
        )
    )

    fig_n.add_hline(y=0, line_width=2, line_color="#4E342E", line_dash="dash")
    max_abs = max(135, max(abs(v) for v in flujos_n + saldos_acum) + 55)
    fig_n.update_layout(
        title=dict(
            text="<b>🧪 Flujo de Nitrógeno por Cultivo y Línea de Acumulación en el Tiempo (kg N/ha)</b>",
            font=dict(size=15, color="#1B5E20"),
        ),
        yaxis=dict(
            title="Kilogramos de Nitrógeno por Hectárea (kg N/ha)",
            range=[-max_abs, max_abs],
            gridcolor="#E0E0E0",
            tickfont=dict(color="#1C2819", size=11),
        ),
        xaxis=dict(
            title="Distribución Temporal (24 Meses / 2 Años)",
            tickfont=dict(color="#1C2819", size=11),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=370,
        margin=dict(l=20, r=20, t=65, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
        font=dict(color="#1C2819"),
    )

    # Gráfico 2: Consumo de Agua en 2 Años calculado con ET_c = Kc * ET_ref y mostrando el ahorro directo
    escenarios_agua = [
        f"Monocultivo 2 Años\n(Repetir {cultivo_actual})",
        "Plan Rotación 2 Años\n(4 Temporadas + Abonos Verdes)",
    ]
    valores_agua = [cuenta["agua_monocultivo_m3_ha"], cuenta["agua_rotacion_m3_ha"]]
    etc_vals = [cuenta["etc_monocultivo_24m_mm"], cuenta["etc_rotacion_24m_mm"]]
    ahorro_m3 = cuenta["ahorro_agua_m3_ha"]
    red_pct = cuenta["reduccion_agua_pct"]

    fig_agua = go.Figure()
    fig_agua.add_trace(
        go.Bar(
            x=escenarios_agua,
            y=valores_agua,
            marker=dict(color=["#E53935", "#0288D1"], line=dict(color="#1C2819", width=1.5)),
            text=[
                f"<b>{valores_agua[0]:,} m³/ha</b><br>(ETc: {etc_vals[0]:.0f} mm)".replace(",", "."),
                f"<b>{valores_agua[1]:,} m³/ha</b><br>(ETc: {etc_vals[1]:.0f} mm)".replace(",", "."),
            ],
            textposition="inside",
            textfont=dict(size=13, color="#FFFFFF"),
            name="Consumo Hídrico 24 Meses",
        )
    )

    fig_agua.add_annotation(
        x=1,
        y=valores_agua[1] + max(valores_agua) * 0.12,
        text=(
            f"<b>💧 REDUCCIÓN DE CONSUMO: -{red_pct:.1f}%</b><br>"
            f"<span style='font-size:12px'>Ahorras <b>{ahorro_m3:,} m³/ha</b> en 2 años (ET = Kc × ET_ref)</span>".replace(",", ".")
        ),
        showarrow=True,
        arrowhead=2,
        arrowcolor="#01579B",
        ax=0,
        ay=-32,
        bgcolor="#E1F5FE",
        bordercolor="#0288D1",
        borderwidth=2,
        borderpad=6,
        font=dict(size=12, color="#01579B"),
    )

    fig_agua.update_layout(
        title=dict(
            text=f"<b>💧 Agua Total en 2 Años (ET = Kc × ET_ref NASA) — Ahorro: -{red_pct:.1f}% ({ahorro_m3:,} m³/ha)</b>".replace(",", "."),
            font=dict(size=15, color="#01579B"),
        ),
        yaxis=dict(
            title="Metros Cúbicos por Hectárea (m³/ha)",
            range=[0, max(valores_agua) * 1.28],
            gridcolor="#E0E0E0",
            tickfont=dict(color="#1C2819", size=11),
        ),
        xaxis=dict(tickfont=dict(color="#1C2819", size=11)),
        height=370,
        margin=dict(l=20, r=20, t=65, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
        showlegend=False,
        font=dict(color="#1C2819"),
    )

    return fig_n, fig_agua


def construir_cuatro_graficos_pronostico(
    df_hist: pd.DataFrame, df_proj: pd.DataFrame, zona: str
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Divide la información agroclimática en 4 gráficos independientes y limpios
    para que las curvas no se solapen y sean fáciles de interpretar por el agricultor.
    """
    df_h = df_hist.tail(240).copy()
    df_p = df_proj.copy()
    fecha_corte = df_h["fecha"].max()

    df_h["hum_raiz_suav"] = df_h["humedad_suelo_raiz"].rolling(7, min_periods=1).mean() * 100
    df_h["temp_suelo_suav"] = (
        df_h["temp_suelo_c"].rolling(7, min_periods=1).mean()
        if "temp_suelo_c" in df_h.columns
        else df_h["temp_media_c"].rolling(7, min_periods=1).mean()
    )
    df_h["temp_aire_suav"] = df_h["temp_media_c"].rolling(7, min_periods=1).mean()

    # 1. Gráfico de Humedad en las Raíces (GWETROOT)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["hum_raiz_suav"],
            mode="lines",
            name="Histórico Reciente",
            line=dict(color="#0277BD", width=3),
            fill="tozeroy",
            fillcolor="rgba(2, 119, 189, 0.14)",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=df_p["humedad_suelo_raiz"] * 100,
            mode="lines",
            name="Proyección 12 Meses",
            line=dict(color="#00897B", width=3, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(0, 137, 123, 0.12)",
        )
    )
    fig1.add_vline(x=fecha_corte, line_width=2, line_dash="dot", line_color="#0B3D91")
    fig1.update_layout(
        title=dict(text=f"<b>💧 1. Agua en Zona de Raíces ({zona}) — Histórico y Proyección (%)</b>", font=dict(size=14, color="#01579B")),
        yaxis=dict(title="Humedad (%)", range=[20, 95], gridcolor="#E0E0E0"),
        xaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=290,
        margin=dict(l=15, r=15, t=55, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
    )

    # 2. Gráfico de Temperatura de la Tierra y del Aire
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=pd.concat([df_h["fecha"], df_p["fecha"]]),
            y=pd.concat([df_h["temp_suelo_suav"], df_p["temp_suelo_c"]]),
            mode="lines",
            name="🤎 Temperatura de la Tierra (TS)",
            line=dict(color="#6D4C41", width=3),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=pd.concat([df_h["fecha"], df_p["fecha"]]),
            y=pd.concat([df_h["temp_aire_suav"], df_p["temp_media_c"]]),
            mode="lines",
            name="☀️ Temperatura del Aire (T2M)",
            line=dict(color="#F57C00", width=2.2, dash="dot"),
        )
    )
    fig2.add_vline(x=fecha_corte, line_width=2, line_dash="dot", line_color="#0B3D91")
    fig2.update_layout(
        title=dict(text=f"<b>🌡️ 2. Temperatura de la Tierra y del Aire ({zona}) — Proyección (°C)</b>", font=dict(size=14, color="#4E342E")),
        yaxis=dict(title="Grados (°C)", gridcolor="#E0E0E0"),
        xaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=290,
        margin=dict(l=15, r=15, t=55, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
    )

    # 3. Gráfico de Evapotranspiración de Referencia (ET_ref) y Lluvia Proyectada
    fig3 = go.Figure()
    et_ref_p = df_p["et_ref_mm"] if "et_ref_mm" in df_p.columns else df_p["evapotranspiracion_mm"] * 1.6
    fig3.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=et_ref_p.rolling(7, min_periods=1).mean(),
            mode="lines",
            name="💨 Evapotranspiración Referencia NASA (ET_ref mm/día)",
            line=dict(color="#D84315", width=3),
        )
    )
    fig3.add_trace(
        go.Bar(
            x=df_p["fecha"],
            y=df_p["precipitacion_mm"],
            name="🌧️ Lluvia Esperada (mm/día)",
            marker=dict(color="#42A5F5", opacity=0.65),
        )
    )
    fig3.update_layout(
        title=dict(text=f"<b>💨 3. Demanda Evaporativa (ET_ref) y Lluvias Proyectadas ({zona})</b>", font=dict(size=14, color="#BF360C")),
        yaxis=dict(title="Milímetros por día (mm/día)", gridcolor="#E0E0E0"),
        xaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=290,
        margin=dict(l=15, r=15, t=55, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
    )

    # 4. Gráfico de Vigor Vegetal (NDVI) y Humedad en la Hoja (NDMI) Sentinel-2
    fig4 = go.Figure()
    if "NDVI" in df_h.columns and "NDVI" in df_p.columns:
        fig4.add_trace(
            go.Scatter(
                x=pd.concat([df_h["fecha"], df_p["fecha"]]),
                y=pd.concat([df_h["NDVI"].rolling(7, min_periods=1).mean(), df_p["NDVI"]]),
                mode="lines",
                name="🌿 Vigor Vegetal (NDVI)",
                line=dict(color="#2E7D32", width=3),
            )
        )
    if "NDMI" in df_h.columns and "NDMI" in df_p.columns:
        fig4.add_trace(
            go.Scatter(
                x=pd.concat([df_h["fecha"], df_p["fecha"]]),
                y=pd.concat([df_h["NDMI"].rolling(7, min_periods=1).mean(), df_p["NDMI"]]),
                mode="lines",
                name="💧 Agua en la Hoja (NDMI)",
                line=dict(color="#0288D1", width=2.5, dash="dash"),
            )
        )
    fig4.add_vline(x=fecha_corte, line_width=2, line_dash="dot", line_color="#0B3D91")
    fig4.update_layout(
        title=dict(text=f"<b>🛰️ 4. Vigor Vegetal (NDVI) y Agua en Hoja (NDMI) — Proyección Sentinel-2</b>", font=dict(size=14, color="#1B5E20")),
        yaxis=dict(title="Índice Óptico (-0.3 a 0.9)", gridcolor="#E0E0E0"),
        xaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=290,
        margin=dict(l=15, r=15, t=55, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBF9F5",
    )

    return fig1, fig2, fig3, fig4


def construir_mapa_interactivo_osm(
    lat_sel: float,
    lon_sel: float,
    ficha_predio: dict,
    cultivo_actual: str,
    primer_recomendado: str = None,
    poligono_guardado: list = None,
    zoom_inicial: int = 12,
) -> folium.Map:
    """Construye el mapa interactivo del Valle Central de Chile con OpenStreetMap +
    capa satelital Esri World Imagery y preserva el nivel de zoom del polígono dibujado
    usando fit_bounds y zoom_preserve para que NO haga zoom-out al cerrar el polígono.
    """
    m = folium.Map(
        location=[lat_sel, lon_sel],
        zoom_start=zoom_inicial,
        tiles=None,
        control_scale=True,
    )

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
        f"<b>🏔️ Pendiente SRTM:</b> {ficha_predio['pendiente_srtm_pct']:.1f}% ({ficha_predio['elevacion_srtm_msnm']} m s.n.m.)<br>"
        f"<b>💧 Humedad Raíz (NASA):</b> {ficha_predio['etiqueta_humedad']}<br>"
        f"<b>🌿 Vigor Sentinel-2:</b> NDVI {ficha_predio['ndvi_actual']:.2f}<br>"
        f"<b>🌾 Cultivo Actual:</b> {icono_act} {cultivo_actual}<br>"
        f"<div style='margin-top:5px; padding:4px 6px; background:#E8F5E9; border-left:4px solid #2E7D32; border-radius:4px;'>"
        f"{texto_rec}</div>"
        f"</div>"
    )

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
        # Ajustar automáticamente el encuadre del mapa a los bordes del polígono para mantener el zoom cercano
        lats_p = [p[0] for p in poligono_guardado]
        lons_p = [p[1] for p in poligono_guardado]
        m.fit_bounds([[min(lats_p), min(lons_p)], [max(lats_p), max(lons_p)]], padding=(28, 28))

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
    lista_cultivos_comerciales = df_hortalizas[~df_hortalizas["es_puente_cobertura"]]["cultivo"].tolist()

    # Inicializar estado de la parcela y zoom en session_state
    if "parcela_lat" not in st.session_state:
        st.session_state["parcela_lat"] = -34.1708
        st.session_state["parcela_lon"] = -70.7444
        st.session_state["tipo_seleccion"] = "Lugar caracterizado: Rancagua"
        st.session_state["area_ha"] = None
        st.session_state["poligono_coords"] = None
        st.session_state["mapa_zoom"] = 12

    ficha_auto = caracterizar_parcela_automatica(
        lat=st.session_state["parcela_lat"],
        lon=st.session_state["parcela_lon"],
        tipo_seleccion=st.session_state["tipo_seleccion"],
        area_ha=st.session_state["area_ha"],
    )
    zona_sel = ficha_auto["zona_cercana"]
    agua_auto = ficha_auto["agua_inferida"]

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
                        Sigue los <b>3 pasos</b> abajo para obtener tu Plan Maestro de 24 Meses (con Grados-Día dinámicos, SRTM y Cuenta Corriente de Nitrógeno y Agua).
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
    # PASO 1 DEL TOUR: UBICAR EL TERRENO EN EL MAPA (CON PRESERVACIÓN DE ZOOM)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🗺️ PASO 1: Ubica tu Terreno o Dibuja el Polígono de tu Parcela</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='tour-step-box' style='border-color:#81D4FA;'>", unsafe_allow_html=True)
    col_ubi_1, col_ubi_2, col_ubi_3 = st.columns([1.25, 1.0, 0.55])
    with col_ubi_1:
        modo_ubicacion = st.radio(
            "1️⃣ Elige cómo marcar tu potrero en el mapa:",
            options=[
                "📐 Dibujar polígono o rectángulo de mi parcela",
                "📍 Hacer un clic rápido sobre mi terreno",
            ],
            horizontal=True,
            index=0,
        )
    es_modo_dibujo = "Dibujar" in modo_ubicacion

    with col_ubi_2:
        lugar_predefinido = st.selectbox(
            "2️⃣ O centra el mapa en una zona agrícola de referencia:",
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
                st.session_state["mapa_zoom"] = 13
                st.rerun()

    with col_ubi_3:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Limpiar Polígono", width="stretch"):
            st.session_state["area_ha"] = None
            st.session_state["poligono_coords"] = None
            st.session_state["tipo_seleccion"] = f"Sector {zona_sel}"
            st.rerun()

    if es_modo_dibujo:
        st.info(
            "📐 **Instrucción para dibujar:** Acércate a tu terreno con el botón **+** del mapa y usa el ícono de **Polígono (pentágono)** o **Rectángulo (cuadrado)** arriba a la izquierda. "
            "Al cerrar la figura, **el mapa mantendrá el zoom exacto sobre tu parcela** sin alejarse."
        )
        objetos_escuchados = ["last_active_drawing"]
    else:
        st.info(
            "📍 **Modo Clic Rápido:** Haz un solo clic sobre cualquier punto de tu terreno en el mapa para evaluarlo al instante."
        )
        objetos_escuchados = ["last_clicked", "last_active_drawing"]
    st.markdown("</div>", unsafe_allow_html=True)

    mapa_osm = construir_mapa_interactivo_osm(
        lat_sel=st.session_state["parcela_lat"],
        lon_sel=st.session_state["parcela_lon"],
        ficha_predio=ficha_auto,
        cultivo_actual="Hortaliza",
        primer_recomendado=None,
        poligono_guardado=st.session_state["poligono_coords"],
        zoom_inicial=st.session_state.get("mapa_zoom", 13),
    )
    salida_mapa = st_folium(
        mapa_osm,
        height=520,
        use_container_width=True,
        key="mapa_tour_paso1",
        returned_objects=objetos_escuchados,
    )

    nueva_coord = extraer_coordenadas_mapa(
        salida_mapa,
        permitir_clic_simple=(not es_modo_dibujo),
        zoom_actual=st.session_state.get("mapa_zoom", 13),
    )
    if nueva_coord is not None:
        n_lat, n_lon, n_tipo, n_area, n_poligono, n_zoom = nueva_coord
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
            st.session_state["mapa_zoom"] = n_zoom
            st.rerun()

    # =========================================================================
    # DIAGNÓSTICO AUTOMÁTICO DEL PUNTO (CATEGORIZADO E INTERPRETADO CON SRTM API)
    # =========================================================================
    interp = ficha_auto["interpretaciones"]
    area_txt = f"📐 Parcela trazada de ~{ficha_auto['area_ha']} ha" if ficha_auto["area_ha"] else f"📍 {ficha_auto['tipo_seleccion']}"

    st.markdown(
        f"<div class='tour-step-box' style='background:linear-gradient(90deg, #E8F5E9 0%, #E1F5FE 100%); border-color:#2E7D32;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem; margin-bottom:0.8rem;'>"
        f"<div>"
        f"<span class='step-badge' style='background:#2E7D32;'>🛰️ DIAGNÓSTICO AUTOMÁTICO E INTERPRETACIÓN DEL TERRENO</span>"
        f"<b style='font-size:1.12rem; color:#1C2819 !important;'>{ficha_auto['nombre_ubicacion']}</b>"
        f"<span style='font-size:0.88rem; color:#4E342E !important; font-weight:700;'> ({ficha_auto['lat']:.4f}°, {ficha_auto['lon']:.4f}° • {area_txt})</span>"
        f"</div>"
        f"<span style='background:#0B3D91; color:#FFFFFF !important; font-size:0.76rem; font-weight:800; padding:0.25rem 0.75rem; border-radius:999px;'>"
        f"📡 Topografía: {ficha_auto['fuente_srtm']}"
        f"</span>"
        f"</div>"
        f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:0.75rem;'>"
        f"<div class='diag-item-card' style='border-left:6px solid {interp['humedad']['color']};'>"
        f"<div style='font-size:0.76rem; font-weight:900; color:{interp['humedad']['color']} !important; text-transform:uppercase;'>💧 Agua en el Suelo (NASA GWETROOT)</div>"
        f"<div style='font-size:1.25rem; font-weight:900; color:#1C2819 !important; margin:0.15rem 0;'>{ficha_auto['humedad_raiz_pct']:.0f}% — {interp['humedad']['escala']}</div>"
        f"<div style='font-size:0.83rem; font-weight:600; color:#37474F !important;'>{interp['humedad']['significado']}</div>"
        f"</div>"
        f"<div class='diag-item-card' style='border-left:6px solid {interp['srtm']['color']};'>"
        f"<div style='font-size:0.76rem; font-weight:900; color:{interp['srtm']['color']} !important; text-transform:uppercase;'>🏔️ Elevación y Pendiente (NASA SRTM)</div>"
        f"<div style='font-size:1.25rem; font-weight:900; color:#1C2819 !important; margin:0.15rem 0;'>{ficha_auto['pendiente_srtm_pct']:.1f}% ({ficha_auto['elevacion_srtm_msnm']} m s.n.m.)</div>"
        f"<div style='font-size:0.8rem; font-weight:800; color:#4E342E !important;'>{ficha_auto['clase_topografica_srtm']} • Erosión {ficha_auto['riesgo_erosion_nivel']}</div>"
        f"<div style='font-size:0.82rem; font-weight:600; color:#37474F !important; margin-top:0.15rem;'>{ficha_auto['escorrentia_srtm']}</div>"
        f"</div>"
        f"<div class='diag-item-card' style='border-left:6px solid {interp['ndvi']['color']};'>"
        f"<div style='font-size:0.76rem; font-weight:900; color:{interp['ndvi']['color']} !important; text-transform:uppercase;'>🌿 Vigor Vegetal (Sentinel-2 NDVI)</div>"
        f"<div style='font-size:1.25rem; font-weight:900; color:#1C2819 !important; margin:0.15rem 0;'>NDVI {ficha_auto['ndvi_actual']:.2f} — {interp['ndvi']['escala']}</div>"
        f"<div style='font-size:0.83rem; font-weight:600; color:#37474F !important;'>{interp['ndvi']['significado']}</div>"
        f"</div>"
        f"<div class='diag-item-card' style='border-left:6px solid {interp['temp_suelo']['color']};'>"
        f"<div style='font-size:0.76rem; font-weight:900; color:{interp['temp_suelo']['color']} !important; text-transform:uppercase;'>🤎 Temperatura de Tierra y Heladas</div>"
        f"<div style='font-size:1.25rem; font-weight:900; color:#1C2819 !important; margin:0.15rem 0;'>{ficha_auto['temp_suelo_c']:.1f} °C — {interp['temp_suelo']['escala']}</div>"
        f"<div style='font-size:0.83rem; font-weight:600; color:#37474F !important;'>{interp['temp_suelo']['significado']} ({interp['heladas']['escala']}).</div>"
        f"</div>"
        f"</div>"
        f"<div style='margin-top:0.7rem; background:#FFFFFF; border-left:5px solid #0B3D91; padding:0.55rem 0.9rem; border-radius:8px; font-size:0.88rem; font-weight:700; color:#1C2819 !important;'>"
        f"🏔️ <b>Recomendación Hidrológica SRTM para tu Pendiente ({ficha_auto['pendiente_srtm_pct']:.1f}%):</b> {ficha_auto['recomendacion_terreno_srtm']}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # PASO 2 DEL TOUR (GRANDE Y FÁCIL DE SELECCIONAR, DIFERENCIADO DEL PASO 3)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-leaf">🌾 PASO 2: Selecciona tu Cultivo Actual (o el Último que Cosechaste en este Potrero)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='tour-step-box' style='border:3px solid #2E7D32; background:#F9FBF9;'>", unsafe_allow_html=True)
    col_p2_sel, col_p2_ficha = st.columns([1.05, 1.25])
    with col_p2_sel:
        st.markdown(
            "<div style='font-size:1.05rem; font-weight:900; color:#1B5E20; margin-bottom:0.35rem;'>"
            "👇 Haz clic aquí para elegir tu hortaliza actual:"
            "</div>",
            unsafe_allow_html=True,
        )
        cultivo_sel = st.selectbox(
            "Cultivo actual o recién cosechado:",
            options=lista_cultivos_comerciales,
            format_func=lambda c: f"{obtener_icono_cultivo(c)}  {c.upper()}",
            index=1 if "Tomate" in lista_cultivos_comerciales else 0,
            label_visibility="collapsed",
        )
        st.caption(
            "💡 Selecciona el cultivo que tienes puesto ahora o el último que sacaste. El modelo buscará familias botánicas distintas para cortar plagas y recuperar los nutrientes extraídos."
        )

    datos_c_act = df_hortalizas[df_hortalizas["cultivo"] == cultivo_sel].iloc[0]
    bal_n_act = int(datos_c_act["balance_nitrogeno_kg_ha"])
    color_n_act = "#2E7D32" if bal_n_act >= 0 else "#C62828"

    with col_p2_ficha:
        st.markdown(
            f"<div style='background:#FFFFFF; border:2.5px solid #81C784; border-left:8px solid #2E7D32; border-radius:14px; padding:0.9rem 1.15rem;'>"
            f"<div style='display:flex; align-items:center; gap:1rem; flex-wrap:wrap;'>"
            f"<span style='font-size:3.5rem; line-height:1;'>{obtener_icono_cultivo(cultivo_sel)}</span>"
            f"<div style='flex:1;'>"
            f"<div style='font-size:0.75rem; font-weight:900; color:#2E7D32 !important; text-transform:uppercase;'>CULTIVO BASE SELECCIONADO EN EL PASO 2</div>"
            f"<div style='font-size:1.45rem; font-weight:900; color:#1C2819 !important;'>{cultivo_sel}</div>"
            f"<div style='font-size:0.88rem; font-weight:700; color:#5D4037 !important; margin-top:0.2rem;'>"
            f"🌿 <b>Familia:</b> {datos_c_act['familia_botanica']} &nbsp;|&nbsp; "
            f"🧪 <b>Efecto Nitrógeno:</b> <span style='color:{color_n_act} !important; font-weight:900;'>{bal_n_act:+} kg N/ha</span> &nbsp;|&nbsp; "
            f"☀️ <b>Requisito Térmico:</b> {int(datos_c_act['gdd_requeridos'])} GDD"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # PASO 3 DEL TOUR (INTEGRA LAS 3 VARIABLES PARA EL SÚPER PLAN DE CULTIVO)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-earth">🎯 PASO 3: Integra las Variables de Prioridad para tu Súper Plan de Rotación</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='tour-step-box' style='border:3px solid #6D4C41; background:#EFEBE9;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1.0rem; font-weight:800; color:#3E2723; margin-bottom:0.5rem;'>"
        "Puedes marcar las <b>3 variables al mismo tiempo</b> (recomendado) para generar el <b>🌟 Súper Plan Integral de Cultivo</b>, o activar solo las que prefieras:"
        "</div>",
        unsafe_allow_html=True,
    )

    c_chk1, c_chk2, c_chk3 = st.columns(3)
    with c_chk1:
        chk_agua = st.checkbox(
            "💧 1. Uso Eficiente de Agua (Ahorro Hídrico ET = Kc × ET_ref)",
            value=True,
        )
    with c_chk2:
        chk_suelo = st.checkbox(
            "🤎 2. Recuperar Suelo y Nitrógeno (Ahorro de Urea + Biofumigación)",
            value=True,
        )
    with c_chk3:
        chk_clima = st.checkbox(
            "☀️ 3. Control de Estrés Térmico y Heladas (Ciclos por Grados-Día GDD)",
            value=True,
        )

    activas = []
    if chk_agua:
        activas.append("💧 Ahorrar agua")
    if chk_suelo:
        activas.append("🤎 Recuperar salud del suelo / nitrógeno")
    if chk_clima:
        activas.append("☀️ Estabilidad térmica (evitar heladas)")

    if len(activas) == 3 or len(activas) == 0:
        prioridad_sel = "🌟 Súper Plan Integral (Todas: Agua + Suelo/Nitrógeno + Estrés Térmico)"
        badge_modo_paso3 = (
            "🌟 <b>SÚPER PLAN INTEGRAL ACTIVO (3 Variables Combinadas):</b> El motor optimiza simultáneamente "
            "el ahorro de agua ($ET_c = K_c \\times ET_{\\text{ref}}$), la fijación biológica de Nitrógeno y la seguridad térmica por Grados-Día (GDD)."
        )
        color_box_p3 = "#2E7D32"
    elif len(activas) == 2:
        prioridad_sel = " + ".join(activas)
        badge_modo_paso3 = f"✅ <b>Plan Combinado Doble Activo:</b> Priorizando <b>{prioridad_sel}</b> junto con disponibilidad hídrica satelital <b>{agua_auto}</b>."
        color_box_p3 = "#0288D1"
    else:
        prioridad_sel = activas[0]
        badge_modo_paso3 = f"🎯 <b>Prioridad Específica Activa:</b> Enfocando la rotación en <b>{prioridad_sel}</b>."
        color_box_p3 = "#6D4C41"

    st.markdown(
        f"<div style='margin-top:0.45rem; background:#FFFFFF; border:2px solid {color_box_p3}; border-left:7px solid {color_box_p3}; padding:0.65rem 1rem; border-radius:10px; font-size:0.92rem; font-weight:700; color:#1C2819 !important;'>"
        f"{badge_modo_paso3}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Ejecutar motor agronómico completo
    resultado = generar_escenarios_rotacion(
        zona=zona_sel,
        nombre_cultivo_actual=cultivo_sel,
        disponibilidad_agua=agua_auto,
        prioridad_usuario=prioridad_sel,
        df_hist_personalizado=ficha_auto["df_interpolado"],
        pendiente_srtm_pct=ficha_auto["pendiente_srtm_pct"],
    )
    ind = resultado["indicadores"]
    plan_24m = resultado["plan_24m"]
    cuenta = plan_24m["cuenta_corriente"]
    alerta_av = plan_24m["alerta_abono_verde"]
    etapas_24m = plan_24m["etapas_24m"]
    icono_actual = obtener_icono_cultivo(cultivo_sel)

    # =========================================================================
    # RESULTADO PRINCIPAL: RESUMEN DEL PLAN MULTI-TEMPORADA 24 MESES
    # =========================================================================
    st.markdown(
        '<div class="section-header header-leaf">🏆 RESULTADO: Tu Súper Plan Multi-Temporada a 2 Años (24 Meses)</div>',
        unsafe_allow_html=True,
    )

    cadena_visual_24m = " ➔ ".join(
        [f"{obtener_icono_cultivo(e['cultivo'])} {e['cultivo']}" for e in etapas_24m]
    )

    st.markdown(
        f"<div class='sequence-master-card'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem; margin-bottom:0.75rem;'>"
        f"<div>"
        f"<span style='background:#2E7D32; color:white !important; padding:0.3rem 0.85rem; border-radius:999px; font-size:0.8rem; font-weight:900;'>"
        f"🏆 SECUENCIA ÓPTIMA DE 24 MESES • CICLOS DINÁMICOS POR GRADOS-DÍA (GDD) EN {zona_sel.upper()}"
        f"</span>"
        f"<h3 style='margin:0.45rem 0 0 0; font-size:1.3rem; font-weight:900; color:#1C2819 !important; line-height:1.35;'>"
        f"{icono_actual} {cultivo_sel} (Actual) ➔ {cadena_visual_24m}"
        f"</h3>"
        f"</div>"
        f"</div>"
        f"<p style='font-size:0.94rem; font-weight:600; color:#263238 !important; margin:0; background:rgba(255,255,255,0.92); padding:0.65rem 0.95rem; border-radius:10px; border-left:5px solid #0288D1;'>"
        f"💡 <b>¿Cómo se adapta este plan a tu zona?</b> Los días y semanas de cada cultivo se calculan dinámicamente con los <b>Grados-Día (GDD)</b> reales de <b>{zona_sel}</b>, alternando <b>{plan_24m['familias_distintas_count']} familias botánicas distintas</b> para cortar plagas de <b>{datos_c_act['familia_botanica']}</b> y manteniendo el suelo 100% cubierto durante los 24 meses."
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # INYECCIÓN AUTOMÁTICA DE ABONO VERDE (DISEÑO AMPLIO, CLARO Y SIN CORTES DE TEXTO)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-leaf">🌿 Inyección Automática de Abonos Verdes (Cultivos Puente de 60 a 80 Días)</div>',
        unsafe_allow_html=True,
    )

    ic_p1 = obtener_icono_cultivo(alerta_av["cultivo_puente_1"])
    ic_p2 = obtener_icono_cultivo(alerta_av["cultivo_puente_2"])

    st.markdown(
        f"<div class='tour-step-box' style='background:#F1F8E9; border:2.5px solid #2E7D32; padding:1.15rem 1.35rem;'>"
        f"<div style='font-size:1.02rem; font-weight:900; color:#1B5E20 !important; margin-bottom:0.4rem;'>"
        f"🔎 ¿Por qué el sistema insertó abonos verdes entre tus cosechas comerciales?"
        f"</div>"
        f"<div style='font-size:0.94rem; font-weight:600; color:#1C2819 !important; line-height:1.45;'>"
        f"Al cruzar la pendiente de tu terreno (<b>{alerta_av['pendiente_srtm_pct']:.1f}% según NASA SRTM</b>), las lluvias proyectadas (<b>{alerta_av['lluvia_ventana_mm']:.0f} mm</b>) "
        f"y los días en que el suelo queda descubierto tras la cosecha de verano, se detectó un <b>Riesgo de Erosión {alerta_av['riesgo_erosion_nivel']}</b>. "
        f"En vez de dejar la tierra vacía, aprovechamos los <b>{alerta_av['gdd_ventana']:.0f} Grados-Día disponibles</b> para sembrar dos cultivos puente rápidos que limpian plagas y fabrican fertilizante natural:"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_av1, col_av2 = st.columns(2)
    with col_av1:
        st.markdown(
            f"<div class='seq-step-box' style='border:2.5px solid #2E7D32; border-left:8px solid #2E7D32; background:#FFFFFF; padding:1.1rem 1.25rem;'>"
            f"<div style='font-size:0.8rem; font-weight:900; color:#2E7D32 !important; text-transform:uppercase;'>"
            f"🌱 PUENTE AÑO 1 • DURACIÓN SEGÚN GDD: {alerta_av['dias_puente_1']} DÍAS ({alerta_av['gdd_puente_1']:.0f} GDD)"
            f"</div>"
            f"<div style='font-size:1.35rem; font-weight:900; color:#1C2819 !important; margin:0.35rem 0;'>"
            f"{ic_p1} {alerta_av['cultivo_puente_1']}"
            f"</div>"
            f"<div style='font-size:0.88rem; font-weight:800; color:#5D4037 !important; margin-bottom:0.45rem;'>"
            f"Aporte directo: {alerta_av['aporte_puente_1']}"
            f"</div>"
            f"<div style='font-size:0.9rem; font-weight:600; color:#263238 !important; background:#F1F8E9; padding:0.65rem 0.85rem; border-radius:10px; line-height:1.4;'>"
            f"✅ <b>Fácil de entender:</b> {alerta_av['razon_puente_1']}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_av2:
        st.markdown(
            f"<div class='seq-step-box' style='border:2.5px solid #558B2F; border-left:8px solid #558B2F; background:#FFFFFF; padding:1.1rem 1.25rem;'>"
            f"<div style='font-size:0.8rem; font-weight:900; color:#558B2F !important; text-transform:uppercase;'>"
            f"🌿 PUENTE AÑO 2 • DURACIÓN SEGÚN GDD: {alerta_av['dias_puente_2']} DÍAS ({alerta_av['gdd_puente_2']:.0f} GDD)"
            f"</div>"
            f"<div style='font-size:1.35rem; font-weight:900; color:#1C2819 !important; margin:0.35rem 0;'>"
            f"{ic_p2} {alerta_av['cultivo_puente_2']}"
            f"</div>"
            f"<div style='font-size:0.88rem; font-weight:800; color:#5D4037 !important; margin-bottom:0.45rem;'>"
            f"Aporte directo: {alerta_av['aporte_puente_2']}"
            f"</div>"
            f"<div style='font-size:0.9rem; font-weight:600; color:#263238 !important; background:#F1F8E9; padding:0.65rem 0.85rem; border-radius:10px; line-height:1.4;'>"
            f"✅ <b>Fácil de entender:</b> {alerta_av['razon_puente_2']}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # =========================================================================
    # CUENTA CORRIENTE DE NITRÓGENO Y AGUA (ENLACE VISUAL UREA <-> AHORRO Y GRÁFICOS SIN TABLA)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🏦 Cuenta Corriente de Nitrógeno y Agua (Impacto Acumulado en 2 Años)</div>',
        unsafe_allow_html=True,
    )

    clp_fmt = f"${cuenta['ahorro_clp_ha']:,}".replace(",", ".")
    extra_parcela_txt = ""
    if ficha_auto.get("area_ha"):
        area_val = float(ficha_auto["area_ha"])
        urea_tot = int(round(cuenta["ahorro_urea_kg_ha"] * area_val))
        clp_tot = f"${int(round(cuenta['ahorro_clp_ha'] * area_val)):,}".replace(",", ".")
        agua_tot = f"{int(round(cuenta['ahorro_agua_m3_ha'] * area_val)):,}".replace(",", ".")
        extra_parcela_txt = (
            f"<div style='margin-top:0.65rem; background:#E8F5E9; border:2px solid #2E7D32; padding:0.55rem 0.9rem; border-radius:10px; font-size:0.92rem; font-weight:800; color:#1B5E20 !important;'>"
            f"📐 <b>Total en tu Parcela Dibujada ({area_val} ha en 2 Años):</b> "
            f"Dejas de comprar <b>{urea_tot:,} kg de Urea</b> (ahorro de <b>{clp_tot} CLP</b>) y ahorras <b>{agua_tot} m³ de agua</b>.".replace(",", ".")
            + "</div>"
        )

    # Tarjeta Superior: Enlace visual directo entre Balance N -> Urea Ahorrada (46% N) -> Ahorro Económico por no comprar esa Urea
    st.markdown(
        f"<div class='tour-step-box' style='background:linear-gradient(135deg, #FFFFFF 0%, #F1F8E9 55%, #FFF8E1 100%); border:3px solid #2E7D32;'>"
        f"<div style='font-size:0.82rem; font-weight:900; color:#1B5E20 !important; text-transform:uppercase; margin-bottom:0.55rem;'>"
        f"🔗 CÁLCULO ENLAZADO DE FERTILIZACIÓN: DEL NITRÓGENO RECUPERADO AL AHORRO EN COMPRA DE INSUMO UREA (46% N)"
        f"</div>"
        f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:0.85rem; align-items:stretch;'>"
        f"<div style='background:#FFFFFF; border:2px solid #2E7D32; border-radius:14px; padding:0.85rem 1rem;'>"
        f"<div style='font-size:0.75rem; font-weight:900; color:#2E7D32 !important;'>1️⃣ BALANCE NETO DE NITRÓGENO (2 AÑOS)</div>"
        f"<div style='font-size:1.6rem; font-weight:900; color:#1B5E20 !important;'>+{cuenta['ganancia_neta_n_kg_ha']} kg N/ha recuperados</div>"
        f"<div style='font-size:0.82rem; font-weight:700; color:#4E342E !important; margin-top:0.2rem;'>"
        f"Tu rotación termina en <b>{cuenta['balance_n_rotacion_kg_ha']:+} kg N/ha</b> frente a <b>{cuenta['balance_n_monocultivo_kg_ha']:+} kg N/ha</b> si repitieras {cultivo_sel} (ver línea azul del gráfico abajo)."
        f"</div>"
        f"</div>"
        f"<div style='background:linear-gradient(90deg, #FFFDE7 0%, #FFF8E1 100%); border:2.5px solid #F57C00; border-radius:14px; padding:0.85rem 1.1rem; grid-column: span 2;'>"
        f"<div style='font-size:0.76rem; font-weight:900; color:#E65100 !important; text-transform:uppercase;'>"
        f"2️⃣ ➔ 3️⃣ EQUIVALENCIA EN FERTILIZANTE UREA (46% N) Y DINERO AHORRADO EN ESE INSUMO"
        f"</div>"
        f"<div style='display:flex; align-items:center; justify-content:space-around; flex-wrap:wrap; gap:0.8rem; margin-top:0.35rem;'>"
        f"<div>"
        f"<div style='font-size:0.78rem; font-weight:800; color:#5D4037 !important;'>Insumo Urea que NO necesitas comprar:</div>"
        f"<div style='font-size:1.6rem; font-weight:900; color:#1C2819 !important;'>🌾 {cuenta['ahorro_urea_kg_ha']:,} kg de Urea / ha</div>".replace(",", ".")
        + f"<div style='font-size:0.82rem; font-weight:700; color:#4E342E !important;'>({cuenta['ganancia_neta_n_kg_ha']} kg N ÷ 0.46 = <b>{cuenta['sacos_urea_25kg_ha']} sacos de 25 kg</b>)</div>"
        f"</div>"
        f"<div style='font-size:2rem; font-weight:900; color:#E65100 !important;'>➔</div>"
        f"<div>"
        f"<div style='font-size:0.78rem; font-weight:800; color:#5D4037 !important;'>Ahorro económico por esos {cuenta['ahorro_urea_kg_ha']:,} kg de Urea:</div>".replace(",", ".")
        + f"<div style='font-size:1.65rem; font-weight:900; color:#E65100 !important;'>💰 {clp_fmt} CLP / ha</div>"
        f"<div style='font-size:0.82rem; font-weight:700; color:#4E342E !important;'>Valorizado a ${cuenta['precio_kg_urea_clp']} CLP/kg Urea (~<b>${cuenta['ahorro_usd_ha']} USD/ha</b>)</div>"
        f"</div>"
        f"</div>"
        f"</div>"
        f"</div>"
        f"{extra_parcela_txt}"
        f"</div>",
        unsafe_allow_html=True,
    )

    fig_flujo_n, fig_agua_24m = crear_figuras_cuenta_corriente_n_y_agua(cuenta, cultivo_sel)
    col_graf_n, col_graf_agua = st.columns([1.22, 0.98])
    with col_graf_n:
        st.plotly_chart(fig_flujo_n, width="stretch")
    with col_graf_agua:
        st.plotly_chart(fig_agua_24m, width="stretch")

    # =========================================================================
    # DETALLES DE CICLOS (FUSIONADO CON LOS DATOS CLAVE E INTERPRETACIÓN POR ESCALA)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-leaf">📅 Detalles de Ciclos e Interpretación por Temporada (Año 1 y Año 2)</div>',
        unsafe_allow_html=True,
    )

    for anio_num, lista_etapas_anio in [(1, plan_24m["etapas_anio1"]), (2, plan_24m["etapas_anio2"])]:
        st.markdown(
            f"<div style='font-size:1.02rem; font-weight:900; color:#1B5E20; margin:0.45rem 0 0.4rem 0;'>"
            f"🗓️ Ciclos del AÑO {anio_num} ({'Meses 1 al 12' if anio_num == 1 else 'Meses 13 al 24'}) — Duración ajustada por Grados-Día (GDD) de {zona_sel}"
            f"</div>",
            unsafe_allow_html=True,
        )
        cols_seq = st.columns(len(lista_etapas_anio))
        for idx_s, etapa in enumerate(lista_etapas_anio):
            ic_s = obtener_icono_cultivo(etapa["cultivo"])
            es_puente = etapa["es_puente_cobertura"]
            c_borde = "#558B2F" if es_puente else ("#0288D1" if idx_s == 1 else "#2E7D32")
            bg_card = "#F1F8E9" if es_puente else "#FFFFFF"

            if etapa["demanda_hidrica"] == "Baja":
                escala_agua = "💧 Baja (Ahorrador)"
            elif etapa["demanda_hidrica"] == "Media":
                escala_agua = "💧💧 Media (Equilibrado)"
            else:
                escala_agua = "💧💧💧 Alta"

            dias_hel = etapa["dias_riesgo_helada"]
            escala_helada = "🟢 Seguro (0 días críticos)" if dias_hel == 0 else f"❄️ Atención ({dias_hel} días fríos)"
            agua_et_fmt = f"{etapa['consumo_agua_m3_ha']:,}".replace(",", ".")

            with cols_seq[idx_s]:
                st.markdown(
                    f"<div class='seq-step-box' style='background:{bg_card}; border-color:{c_borde}; border-left: 7px solid {c_borde};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<span style='font-size:0.75rem; font-weight:900; color:{c_borde} !important; text-transform:uppercase;'>"
                    f"ETAPA {etapa['orden']} • {etapa['etiqueta_periodo']}"
                    f"</span>"
                    f"<span style='background:{c_borde}; color:#FFFFFF !important; font-size:0.72rem; font-weight:800; padding:2px 8px; border-radius:999px;'>"
                    f"{'ABONO VERDE PUENTE' if es_puente else 'CULTIVO COMERCIAL'}"
                    f"</span>"
                    f"</div>"
                    f"<div style='display:flex; align-items:center; gap:0.65rem; margin:0.45rem 0;'>"
                    f"<span style='font-size:2.6rem; line-height:1;'>{ic_s}</span>"
                    f"<div>"
                    f"<div style='font-size:1.25rem; font-weight:900; color:#1C2819 !important;'>{etapa['cultivo']}</div>"
                    f"<div style='font-size:0.82rem; font-weight:700; color:#5D4037 !important;'>🌿 {etapa['familia_botanica']}</div>"
                    f"</div>"
                    f"</div>"
                    f"<div class='visual-chips-grid'>"
                    f"<div class='v-chip chip-leaf'>"
                    f"<span class='label'>Tiempo a Cosecha (GDD)</span>"
                    f"<span class='val'>🗓️ {etapa['ciclo_dinamico_dias']} días ({etapa['duracion_semanas']} sem)</span>"
                    f"</div>"
                    f"<div class='v-chip chip-water'>"
                    f"<span class='label'>Consumo Agua (ET=Kc×ET_ref)</span>"
                    f"<span class='val'>{escala_agua} • {agua_et_fmt} m³/ha</span>"
                    f"</div>"
                    f"<div class='v-chip chip-earth'>"
                    f"<span class='label'>Aporte Nitrógeno / Tierra</span>"
                    f"<span class='val'>{etapa['categoria_nitrogeno']}</span>"
                    f"</div>"
                    f"<div class='v-chip chip-sun'>"
                    f"<span class='label'>Resistencia al Frío</span>"
                    f"<span class='val'>{escala_helada}</span>"
                    f"</div>"
                    f"</div>"
                    f"<div class='short-tip' style='border-left-color:{c_borde};'>"
                    f"☀️ <b>Ajuste Térmico Zonal:</b> {etapa['explicacion_gdd']}<br>"
                    f"🌱 <b>Beneficio clave frente a repetir {cultivo_sel}:</b> {etapa['descripcion_beneficio']}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # CARTAS GANTT MULTI-TEMPORADA (24 MESES SEPARADOS POR AÑO EN 2 TABLAS DE 12 MESES)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-earth">🗓️ Carta Gantt Multi-Temporada (24 Meses: Año 1 y Año 2 Separados por Tabla)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(construir_leyenda_gantt_cuadrados_html(), unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:1.04rem; font-weight:900; color:#0B3D91; margin:0.35rem 0 0.2rem 0;'>"
        "📅 Tabla Carta Gantt — AÑO 1 (Meses 1 al 12 • Semanas 1 a 4 de cada mes en la parte superior)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        construir_figura_gantt_plotly(plan_24m, modo_vista="anio_1"),
        width="stretch",
    )

    st.markdown(
        "<div style='font-size:1.04rem; font-weight:900; color:#1B5E20; margin:0.65rem 0 0.2rem 0;'>"
        "📅 Tabla Carta Gantt — AÑO 2 (Meses 13 al 24 • Semanas 1 a 4 de cada mes en la parte superior)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        construir_figura_gantt_plotly(plan_24m, modo_vista="anio_2"),
        width="stretch",
    )

    # =========================================================================
    # SECCIÓN 2: MONITOREO SATELITAL (TÍTULO LIMPIO Y TRADUCCIÓN SENCILLA DE CADA VALOR)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🛰️ Monitoreo Satelital</div>',
        unsafe_allow_html=True,
    )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        val_ar = round(ind["humedad_raiz_actual"] * 100, 1)
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Agua en Raíces",
                icono="💧",
                valor=val_ar,
                sufijo="%",
                min_val=0,
                max_val=100,
                color_barra="#0288D1",
                color_fondo_rango="#E1F5FE",
                subtitulo=f"NASA GWETROOT (Proyección: {ind['humedad_raiz_proj']*100:.0f}%)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("agua_raices", val_ar)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g2:
        val_ts = ind["temp_suelo_actual_c"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Temp. de la Tierra",
                icono="🤎",
                valor=val_ts,
                sufijo="°C",
                min_val=0,
                max_val=35,
                color_barra="#6D4C41",
                color_fondo_rango="#EFEBE9",
                subtitulo=f"NASA TS (Proyección: {ind['temp_suelo_proj_c']:.1f} °C)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("temp_tierra", val_ts)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g3:
        val_gdd = ind["gdd_proj_120"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Calor Acumulado",
                icono="☀️",
                valor=val_gdd,
                sufijo="GDD",
                min_val=0,
                max_val=1800,
                color_barra="#2E7D32",
                color_fondo_rango="#E8F5E9",
                subtitulo="Grados-Día (Próximos 120 días)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("calor_gdd", val_gdd)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g4:
        val_ev = ind["evap_diaria_actual_mm"]
        et_ref_d = ind["et_ref_diaria_proj_mm"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Evaporación Diaria",
                icono="💨",
                valor=val_ev,
                sufijo="mm/d",
                min_val=0,
                max_val=6,
                color_barra="#F57C00",
                color_fondo_rango="#FFF8E1",
                subtitulo=f"NASA EVLAND (ET_ref: {et_ref_d:.1f} mm/d)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("evaporacion", val_ev, et_ref_d)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    g5, g6, g7, g8 = st.columns(4)
    with g5:
        val_rad = ind["radiacion_actual"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Luz del Sol",
                icono="🔆",
                valor=val_rad,
                sufijo="MJ",
                min_val=0,
                max_val=35,
                color_barra="#FBC02D",
                color_fondo_rango="#FFFDE7",
                subtitulo="Radiación Solar (ALLSKY_SFC_SW_DWN)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("luz_solar", val_rad)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g6:
        val_rh = ind["humedad_rel_actual_pct"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Humedad del Aire",
                icono="☁️",
                valor=val_rh,
                sufijo="%",
                min_val=0,
                max_val=100,
                color_barra="#039BE5",
                color_fondo_rango="#E1F5FE",
                subtitulo="NASA RH2M (Alerta hongos >75%)",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("humedad_aire", val_rh)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g7:
        val_lluv = ind["precip_ultimo_anio_mm"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Lluvia del Año",
                icono="🌧️",
                valor=val_lluv,
                sufijo="mm",
                min_val=0,
                max_val=1200,
                color_barra="#1565C0",
                color_fondo_rango="#E3F2FD",
                subtitulo=f"Promedio normal: {ind['precip_media_anual_mm']:.0f} mm/año",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("lluvia", val_lluv)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    with g8:
        val_hel = ind["dias_riesgo_cultivo_proj"]
        st.plotly_chart(
            crear_medidor_variable(
                titulo="Días de Helada",
                icono="❄️",
                valor=val_hel,
                sufijo="días",
                min_val=0,
                max_val=30,
                color_barra="#00ACC1",
                color_fondo_rango="#E0F7FA",
                subtitulo="Proyección Próximos 12 Meses",
            ),
            width="stretch",
        )
        t_niv, t_msg, t_col = traducir_valor_monitoreo("heladas", val_hel)
        st.markdown(
            f"<div class='short-tip' style='border-left-color:{t_col}; margin-top:-0.5rem;'><b>{t_niv}:</b> {t_msg}</div>",
            unsafe_allow_html=True,
        )

    # =========================================================================
    # ÍNDICES ÓPTICOS DE VEGETACIÓN Y SUELOS (EXPLICACIÓN SENCILLA + PALABRA COMPLETA PROYECCIÓN)
    # =========================================================================
    st.markdown(
        '<div style="font-size:1.05rem; font-weight:900; color:#1B5E20; margin:0.9rem 0 0.5rem 0;">🛰️ Índices Ópticos de Vegetación y Suelos — Copernicus Sentinel-2 L2A (Explicación Sencilla)</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"<div class='sentinel-badge-card' style='border-left:6px solid #2E7D32;'>"
            f"<div style='font-size:0.78rem; font-weight:900; color:#2E7D32 !important;'>🌿 VIGOR VEGETATIVO (NDVI)</div>"
            f"<div style='font-size:1.45rem; font-weight:900; color:#1C2819 !important; margin:0.2rem 0;'>{ind['ndvi_actual']:.2f} <span style='font-size:0.84rem; font-weight:800; color:#2E7D32 !important;'>(Proyección 12 meses: {ind['ndvi_proj_12m']:.2f})</span></div>"
            f"<div style='font-size:0.82rem; font-weight:600; color:#37474F !important;'><b>¿Qué es?</b> Mide qué tan verdes y activas están las plantas (de 0 a 1). Sobre 0.50 indica plantas sanas creciendo con fuerza.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='sentinel-badge-card' style='border-left:6px solid #558B2F;'>"
            f"<div style='font-size:0.78rem; font-weight:900; color:#558B2F !important;'>🌱 DENSIDAD DE FOLLAJE (EVI)</div>"
            f"<div style='font-size:1.45rem; font-weight:900; color:#1C2819 !important; margin:0.2rem 0;'>{ind['evi_actual']:.2f} <span style='font-size:0.84rem; font-weight:800; color:#558B2F !important;'>(Alta Biomasa)</span></div>"
            f"<div style='font-size:0.82rem; font-weight:600; color:#37474F !important;'><b>¿Qué es?</b> Mide la cantidad de hojas y tallos cuando el cultivo ya está grande y tupido, descontando el brillo de la tierra.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='sentinel-badge-card' style='border-left:6px solid #0288D1;'>"
            f"<div style='font-size:0.78rem; font-weight:900; color:#0288D1 !important;'>💧 AGUA DENTRO DE LA HOJA (NDMI)</div>"
            f"<div style='font-size:1.45rem; font-weight:900; color:#1C2819 !important; margin:0.2rem 0;'>{ind['ndmi_actual']:+.2f} <span style='font-size:0.84rem; font-weight:800; color:#0288D1 !important;'>(Proyección 12 meses: {ind['ndmi_proj_12m']:+.2f})</span></div>"
            f"<div style='font-size:0.82rem; font-weight:600; color:#37474F !important;'><b>¿Qué es?</b> Detecta si el cultivo tiene sed antes de que se marchite. Valores positivos indican hojas bien hidratadas.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"<div class='sentinel-badge-card' style='border-left:6px solid #6D4C41;'>"
            f"<div style='font-size:0.78rem; font-weight:900; color:#6D4C41 !important;'>🤎 RIESGO DE SUELO DESNUDO</div>"
            f"<div style='font-size:1.45rem; font-weight:900; color:#1C2819 !important; margin:0.2rem 0;'>{ind['suelo_desnudo_pct']:.0f}% <span style='font-size:0.84rem; font-weight:800; color:#2E7D32 !important;'>(Proyección con Plan: 0%)</span></div>"
            f"<div style='font-size:0.82rem; font-weight:600; color:#37474F !important;'><b>¿Qué es?</b> Porcentaje del año en que la tierra queda pelada sin plantas ({ind['suelo_desnudo_dias_anual']} días/año), expuesta a secarse y erosionarse.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # =========================================================================
    # SECCIÓN 3: SALUD DE TU TIERRA Y PRONÓSTICO AGROCLIMÁTICO (4 GRÁFICOS SEPARADOS SIN SOLAPAR)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-earth">🤎 Salud de tu Tierra y Pronóstico Agroclimático (Gráficos Separados para Fácil Lectura)</div>',
        unsafe_allow_html=True,
    )

    fig_p1, fig_p2, fig_p3, fig_p4 = construir_cuatro_graficos_pronostico(
        resultado["df_historico"], resultado["df_proyeccion"], zona_sel
    )
    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        st.plotly_chart(fig_p1, width="stretch")
        st.plotly_chart(fig_p3, width="stretch")
    with col_pr2:
        st.plotly_chart(fig_p2, width="stretch")
        st.plotly_chart(fig_p4, width="stretch")

    # =========================================================================
    # SECCIÓN 4: TABLAS TÉCNICAS COMPLETAS (MANTENIDAS COMO ESTÁN)
    # =========================================================================
    st.markdown(
        '<div class="section-header header-water">🔬 Tablas Técnicas Completas (Catálogo Agronómico y Datos Satelitales NASA POWER + Sentinel-2)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.95rem; font-weight:900; color:#1B5E20; margin:0.3rem 0;'>📊 Tabla Técnica de Cultivos (Kc, GDD, Heladas, Nitrógeno y Agua)</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_hortalizas, width="stretch", hide_index=True)

    st.markdown(
        "<div style='font-size:0.95rem; font-weight:900; color:#01579B; margin:0.6rem 0 0.3rem 0;'>🛰️ Datos Recientes del Satélite NASA POWER + Copernicus Sentinel-2 (Últimos 90 días)</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        resultado["df_historico"].tail(90).sort_values("fecha", ascending=False),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
