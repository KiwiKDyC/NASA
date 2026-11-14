import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from descargar_datos import ZONAS_CHILE
from modelo_rotacion import (
    cargar_catalogo_hortalizas,
    generar_escenarios_rotacion,
)

st.set_page_config(
    page_title="AgroRotación NASA | Valle Central de Chile",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS responsivos para escritorio y móvil
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #065f46 100%);
        color: #f8fafc;
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin-top: 0.35rem;
    }
    .scenario-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        height: 100%;
    }
    .scenario-winner {
        border: 2.5px solid #10b981;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.14);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    .crop-name {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.2rem 0;
    }
    .family-tag {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.86rem;
        padding: 0.32rem 0;
        border-bottom: 1px dashed #e2e8f0;
        color: #1e293b;
    }
    @media (prefers-color-scheme: dark) {
        .scenario-card {
            background-color: #1e293b;
            border-color: #334155;
        }
        .crop-name {
            color: #f8fafc;
        }
        .family-tag {
            color: #94a3b8;
        }
        .metric-row {
            color: #e2e8f0;
            border-bottom-color: #334155;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def construir_grafico_clima_suelo(df_hist: pd.DataFrame, df_proj: pd.DataFrame, zona: str) -> go.Figure:
    """Crea gráfico interactivo de doble panel: Humedad de Suelo (GWETROOT/GWETPROF/GWETTOP)
    y Temperaturas (Aire T2M vs Suelo TS) con últimos 12 meses históricos + 6 meses proyectados.
    """
    df_h = df_hist.tail(365).copy()
    df_p = df_proj.copy()

    # Suavizado móvil de 7 días para visualización limpia en histórico
    df_h["hum_raiz_suav"] = df_h["humedad_suelo_raiz"].rolling(7, min_periods=1).mean()
    df_h["hum_sup_suav"] = df_h["humedad_suelo_sup"].rolling(7, min_periods=1).mean()
    df_h["temp_media_suav"] = df_h["temp_media_c"].rolling(7, min_periods=1).mean()
    if "temp_suelo_c" in df_h.columns:
        df_h["temp_suelo_suav"] = df_h["temp_suelo_c"].rolling(7, min_periods=1).mean()
    else:
        df_h["temp_suelo_suav"] = df_h["temp_media_suav"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"💧 Humedad del Suelo NASA POWER en {zona} (Zona Radicular GWETROOT y Superficial GWETTOP)",
            f"🌡️ Dinámica Térmica en {zona} (Temperatura Media 2m T2M y Temperatura de Suelo TS)",
        ),
        row_heights=[0.52, 0.48],
    )

    # --- PANEL 1: HUMEDAD DEL SUELO ---
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["hum_raiz_suav"] * 100,
            mode="lines",
            name="Humedad Raíz Histórica (GWETROOT %)",
            line=dict(color="#0284c7", width=2.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["hum_sup_suav"] * 100,
            mode="lines",
            name="Humedad Superficial Histórica (GWETTOP %)",
            line=dict(color="#38bdf8", width=1.4, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=df_p["humedad_suelo_raiz"] * 100,
            mode="lines",
            name="Proyección 6 Meses Humedad Raíz (%)",
            line=dict(color="#10b981", width=3.0, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # --- PANEL 2: TEMPERATURAS ---
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["temp_media_suav"],
            mode="lines",
            name="Temp. Aire 2m Histórica (T2M °C)",
            line=dict(color="#f59e0b", width=2.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_h["fecha"],
            y=df_h["temp_suelo_suav"],
            mode="lines",
            name="Temp. Suelo Histórica (TS °C)",
            line=dict(color="#b45309", width=1.6, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p["fecha"],
            y=df_p["temp_media_c"],
            mode="lines",
            name="Proyección 6 Meses Temp. Aire (°C)",
            line=dict(color="#ef4444", width=2.8, dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Línea vertical divisoria entre Histórico y Proyección
    fecha_corte = df_h["fecha"].max()
    fig.add_vline(
        x=fecha_corte,
        line_width=1.8,
        line_dash="dashdot",
        line_color="#64748b",
    )

    fig.update_yaxes(title_text="Saturación (%)", range=[10, 100], row=1, col=1)
    fig.update_yaxes(title_text="Temperatura (°C)", row=2, col=1)
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def main():
    df_hortalizas = cargar_catalogo_hortalizas()
    lista_cultivos = df_hortalizas["cultivo"].tolist()

    # --- BARRA LATERAL (FILTROS DEL AGRICULTOR) ---
    with st.sidebar:
        st.markdown("## 🛰️ Configuración Predial")
        st.caption("Datos agroclimáticos satelitales **NASA POWER (2021-2026)** | Modo 100% Offline")

        zona_sel = st.selectbox(
            "📍 Comuna / Zona Agroclimática",
            options=list(ZONAS_CHILE.keys()),
            format_func=lambda z: f"{z} (Región de {ZONAS_CHILE[z]['region']})",
            index=0,
        )

        cultivo_sel = st.selectbox(
            "🌱 Cultivo Actual o Recién Cosechado",
            options=lista_cultivos,
            index=1 if "Tomate" in lista_cultivos else 0,
        )

        agua_sel = st.select_slider(
            "💧 Disponibilidad de Agua en el Predio",
            options=["Baja", "Media", "Alta"],
            value="Baja",
        )

        prioridad_sel = st.radio(
            "🎯 Prioridad Productiva para la Próxima Rotación",
            options=[
                "Ahorrar agua",
                "Recuperar salud del suelo / nitrógeno",
                "Estabilidad térmica",
            ],
            index=0,
        )

        st.divider()
        info_z = ZONAS_CHILE[zona_sel]
        st.markdown(
            f"""
            **Ficha Satelital ({zona_sel}):**
            - **Coordenadas:** `{info_z['lat']}°, {info_z['lon']}°`
            - **Región:** {info_z['region']}
            - **Fuente:** NASA LaRC POWER Agroclimatology
            """
        )

    # --- EJECUTAR MOTOR DE ANÁLISIS Y ROTACIÓN ---
    resultado = generar_escenarios_rotacion(
        zona=zona_sel,
        nombre_cultivo_actual=cultivo_sel,
        disponibilidad_agua=agua_sel,
        prioridad_usuario=prioridad_sel,
    )
    ind = resultado["indicadores"]
    cultivo_act = resultado["cultivo_actual"]
    ganador = resultado["recomendacion_principal"]

    # --- ENCABEZADO PRINCIPAL ---
    st.markdown(
        f"""
        <div class="hero-banner">
            <p class="hero-title">🛰️ AgroRotación Chile — Sistema de Apoyo a la Decisión con Datos NASA</p>
            <p class="hero-sub">
                Recomendador inteligente de rotación de hortalizas en el Valle Central |
                <strong>Zona:</strong> {zona_sel} ({ZONAS_CHILE[zona_sel]['region']}) •
                <strong>Cultivo Base:</strong> {cultivo_sel} ({cultivo_act['familia_botanica']}) •
                <strong>Recomendación Óptima:</strong> {ganador['cultivo_recomendado']} ({resultado['estrategia_ganadora']})
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- FILA DE KPIs SATELITALES NASA POWER ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="💧 Humedad Raíz (GWETROOT)",
            value=f"{ind['humedad_raiz_actual']*100:.1f}%",
            delta=f"{ind['delta_humedad_raiz']*100:+.1f}% proy. 6m",
        )
    with col2:
        st.metric(
            label="☀️ GDD Proyectados (120d)",
            value=f"{ind['gdd_proj_120']:.0f} GDD",
            delta=f"Base {cultivo_act['t_base_gdd']} °C",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="🌡️ Temp. Suelo (NASA TS)",
            value=f"{ind['temp_suelo_actual_c']:.1f} °C",
            delta=f"Proy: {ind['temp_suelo_proj_c']:.1f} °C",
            delta_color="off",
        )
    with col4:
        st.metric(
            label="🌧️ Precipitación Anual",
            value=f"{ind['precip_ultimo_anio_mm']:.0f} mm",
            delta=f"{ind['anomalia_precip_pct']:+.1f}% vs media",
        )
    with col5:
        st.metric(
            label="❄️ Riesgo Heladas (6m)",
            value=f"{ind['dias_riesgo_cultivo_proj']} días",
            delta=f"Umbral {cultivo_act['t_min_helada']} °C",
            delta_color="inverse",
        )

    # --- ALERTAS AGRONÓMICAS ---
    for alerta in resultado["alertas"]:
        if alerta["tipo"] == "warning":
            st.warning(f"**{alerta['titulo']}:** {alerta['mensaje']}", icon="⚠️")
        elif alerta["tipo"] == "info":
            st.info(f"**{alerta['titulo']}:** {alerta['mensaje']}", icon="ℹ️")
        else:
            st.success(f"**{alerta['titulo']}:** {alerta['mensaje']}", icon="✅")

    # --- TARJETAS COMPARATIVAS DE LOS 3 ESCENARIOS ---
    st.subheader("🔄 Comparativa de 3 Escenarios de Rotación para el Próximo Ciclo")
    cols_esc = st.columns(3)

    for idx, esc in enumerate(resultado["escenarios"]):
        es_ganador = esc["titulo"] == ganador["titulo"]
        clase_extra = "scenario-winner" if es_ganador else ""
        etiqueta_ganador = " ⭐ RECOMENDADO PARA TU PRIORIDAD" if es_ganador else ""
        det = esc["detalles"]

        puntos_html = "".join([f"<li style='margin-bottom:4px;'>{p}</li>" for p in esc["puntos_clave"]])

        with cols_esc[idx]:
            st.markdown(
                f"""
                <div class="scenario-card {clase_extra}">
                    <span class="badge-pill" style="background-color: {esc['color_badge']};">
                        {esc['nivel_riesgo']}{etiqueta_ganador}
                    </span>
                    <div style="font-size:0.85rem; font-weight:700; color:#64748b;">{esc['titulo']}</div>
                    <div class="crop-name">{esc['cultivo_recomendado']}</div>
                    <div class="family-tag">🌿 Familia: {esc['familia_botanica']} • 🗓️ {esc['temporada_optima']}</div>
                    <div class="metric-row">
                        <span>Índice de Viabilidad Agro-NASA:</span>
                        <strong>{esc['puntaje_viabilidad']:.0f} / 100</strong>
                    </div>
                    <div class="metric-row">
                        <span>Demanda Hídrica (Coef. Kc):</span>
                        <strong>{esc['demanda_hidrica']} (Kc = {esc['kc_promedio']})</strong>
                    </div>
                    <div class="metric-row">
                        <span>Efecto en el Suelo:</span>
                        <strong>{esc['aporte_suelo']}</strong>
                    </div>
                    <div class="metric-row">
                        <span>Cumplimiento Térmico (GDD):</span>
                        <strong>{det['cumplimiento_gdd_pct']}% ({det['gdd_disponibles']:.0f}/{det['gdd_requeridos']})</strong>
                    </div>
                    <div class="metric-row">
                        <span>Días con Riesgo de Helada:</span>
                        <strong>{det['dias_riesgo_helada']} días (en {esc['ciclo_dias']}d)</strong>
                    </div>
                    <p style="font-size:0.85rem; margin-top:0.75rem; color:#334155;">
                        {esc['justificacion']}
                    </p>
                    <ul style="font-size:0.82rem; padding-left:1.1rem; margin-bottom:0;">
                        {puntos_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- GRÁFICO INTERACTIVO PLOTLY Y TABLA EXPLORATORIA ---
    tab_grafico, tab_catalogo, tab_datos_nasa = st.tabs([
        "📈 Evolución Histórica y Proyección NASA (6 Meses)",
        "🌱 Catálogo Agronómico de Hortalizas",
        "🛰️ Datos Crudos NASA POWER Extraídos",
    ])

    with tab_grafico:
        fig = construir_grafico_clima_suelo(
            resultado["df_historico"], resultado["df_proyeccion"], zona_sel
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**Nota metodológica:** La línea punteada proyecta los próximos 180 días combinando tendencia "
            "secular multianual (2021-2026) y armónicos estacionales de Fourier calibrados con los últimos 14 días observados por NASA POWER."
        )

    with tab_catalogo:
        st.markdown("#### Matriz de Requerimientos Agronómicos y Rotación Fitosanitaria")
        st.dataframe(
            df_hortalizas,
            use_container_width=True,
            hide_index=True,
        )

    with tab_datos_nasa:
        st.markdown(f"#### Últimos registros diarios extraídos de NASA POWER para **{zona_sel}**")
        st.dataframe(
            resultado["df_historico"].tail(60).sort_values("fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
