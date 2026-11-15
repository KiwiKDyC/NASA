from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path("data")


def cargar_datos_zona(zona: str) -> pd.DataFrame:
    """Carga el CSV local de NASA POWER para la zona seleccionada."""
    nombre_archivo = f"nasa_{zona.lower()}.csv"
    ruta = DATA_DIR / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {ruta}. Ejecuta primero descargar_datos.py"
        )
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def cargar_catalogo_hortalizas() -> pd.DataFrame:
    """Carga la base de datos agronómica de hortalizas."""
    ruta = DATA_DIR / "base_hortalizas.csv"
    if not ruta.exists():
        from crear_base_hortalizas import generar_catalogo_hortalizas
        return generar_catalogo_hortalizas()
    return pd.read_csv(ruta)


def proyectar_clima_6_meses(df_hist: pd.DataFrame, dias_horizonte: int = 365) -> pd.DataFrame:
    """Proyecta temperatura media, mínima, del suelo, evapotranspiración, humedad radicular
    e índices espectrales Sentinel-2 (NDVI, EVI, NDMI, SUELO_DESNUDO) para los próximos 12 meses
    (365 días / 48 semanas) usando regresión lineal armónica (tendencia + estacionalidad de Fourier).
    """
    df = df_hist.copy()
    fecha_min = df["fecha"].min()
    t = (df["fecha"] - fecha_min).dt.days.to_numpy(dtype=float)
    doy = df["fecha"].dt.dayofyear.to_numpy(dtype=float)

    # Matriz de diseño: [1, tendencia_lineal, sin(2pi*doy), cos(2pi*doy), sin(4pi*doy), cos(4pi*doy)]
    def construir_matriz(t_arr: np.ndarray, doy_arr: np.ndarray) -> np.ndarray:
        w = 2.0 * np.pi * doy_arr / 365.25
        return np.column_stack([
            np.ones_like(t_arr),
            t_arr / 365.25,
            np.sin(w),
            np.cos(w),
            np.sin(2 * w),
            np.cos(2 * w),
        ])

    X_train = construir_matriz(t, doy)

    ultima_fecha = df["fecha"].max()
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=dias_horizonte,
        freq="D",
    )
    t_fut = (fechas_futuras - fecha_min).days.to_numpy(dtype=float)
    doy_fut = fechas_futuras.dayofyear.to_numpy(dtype=float)
    X_fut = construir_matriz(t_fut, doy_fut)

    df_proj = pd.DataFrame({"fecha": fechas_futuras})

    columnas_a_proyectar = [
        "temp_media_c",
        "temp_max_c",
        "temp_min_c",
        "temp_suelo_c",
        "humedad_suelo_raiz",
        "humedad_suelo_sup",
        "humedad_suelo_prof",
        "evapotranspiracion_mm",
        "radiacion_solar",
        "NDVI",
        "EVI",
        "NDMI",
    ]

    for col in columnas_a_proyectar:
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            coef, _, _, _ = np.linalg.lstsq(X_train, y, rcond=None)
            pred = X_fut @ coef

            # Ajustar suavemente el empalme con el promedio de los últimos 14 días reales
            ultimo_nivel_real = df[col].tail(14).mean()
            desfase_inicial = (ultimo_nivel_real - pred[0]) * np.exp(-np.arange(dias_horizonte) / 25.0)
            pred_ajustada = pred + desfase_inicial

            if "humedad_suelo" in col:
                pred_ajustada = np.clip(pred_ajustada, 0.10, 0.95)
            elif col == "evapotranspiracion_mm":
                pred_ajustada = np.clip(pred_ajustada, 0.05, 10.0)
            elif col == "NDVI":
                pred_ajustada = np.clip(pred_ajustada, 0.08, 0.88)
            elif col == "EVI":
                pred_ajustada = np.clip(pred_ajustada, 0.05, 0.78)
            elif col == "NDMI":
                pred_ajustada = np.clip(pred_ajustada, -0.35, 0.65)

            usa_3_dec = ("humedad_suelo" in col) or (col in ["NDVI", "EVI", "NDMI"])
            df_proj[col] = np.round(pred_ajustada, 3 if usa_3_dec else 2)

    if "NDVI" in df_proj.columns and "humedad_suelo_sup" in df_proj.columns:
        df_proj["SUELO_DESNUDO"] = np.where(
            (df_proj["NDVI"] < 0.28) | ((df_proj["NDVI"] < 0.34) & (df_proj["humedad_suelo_sup"] < 0.28)),
            1,
            0,
        )

    # Estimar precipitación proyectada por climatología diaria histórica del lugar
    df["doy"] = df["fecha"].dt.dayofyear
    precip_clim = df.groupby("doy")["precipitacion_mm"].mean().to_dict()
    df_proj["precipitacion_mm"] = [
        round(precip_clim.get(d, 0.5), 2) for d in fechas_futuras.dayofyear
    ]
    df_proj["tipo_dato"] = "Proyección 12 Meses (IA/Estacional)"
    return df_proj


proyectar_clima_12_meses = proyectar_clima_6_meses


def calcular_indicadores_agronomicos(
    df_hist: pd.DataFrame,
    df_proj: pd.DataFrame,
    t_base_cultivo: float = 8.0,
    t_min_helada_cultivo: float = 0.0,
) -> Dict[str, Any]:
    """Calcula indicadores agroclimáticos históricos y proyectados a partir de NASA POWER."""
    ultimos_365 = df_hist.tail(365).copy()

    # 1. Grados-Día de Crecimiento (GDD) en últimos 120 días y proyectados próximos 120 días
    ventana_120_hist = df_hist.tail(120)
    gdd_hist_120 = np.maximum(0.0, ventana_120_hist["temp_media_c"] - t_base_cultivo).sum()

    ventana_120_proj = df_proj.head(120)
    gdd_proj_120 = np.maximum(0.0, ventana_120_proj["temp_media_c"] - t_base_cultivo).sum()

    # 2. Riesgo de heladas (histórico anual y proyectado 6 meses)
    dias_helada_anual = int((ultimos_365["temp_min_c"] <= 0.0).sum())
    dias_riesgo_cultivo_proj = int((df_proj["temp_min_c"] <= t_min_helada_cultivo).sum())

    # 3. Anomalía de precipitación (últimos 12 meses vs promedio histórico de toda la serie)
    precip_ultimo_anio = ultimos_365["precipitacion_mm"].sum()
    anios_totales = max(1.0, len(df_hist) / 365.25)
    precip_media_anual = df_hist["precipitacion_mm"].sum() / anios_totales
    anomalia_precip_pct = (
        ((precip_ultimo_anio - precip_media_anual) / precip_media_anual) * 100.0
        if precip_media_anual > 0
        else 0.0
    )

    # 4. Humedad radicular (GWETROOT) y tendencia a 6 meses
    humedad_raiz_actual = float(df_hist.tail(30)["humedad_suelo_raiz"].mean())
    humedad_raiz_proj = float(df_proj["humedad_suelo_raiz"].mean())
    delta_humedad_raiz = humedad_raiz_proj - humedad_raiz_actual

    # 5. Nuevas variables NASA: Temperatura de suelo (TS), Evapotranspiración (EVLAND), Humedad profunda (GWETPROF), Superficial (GWETTOP), RH2M y Radiación
    temp_suelo_actual = (
        float(df_hist.tail(30)["temp_suelo_c"].mean())
        if "temp_suelo_c" in df_hist.columns
        else float(df_hist.tail(30)["temp_media_c"].mean())
    )
    temp_suelo_proj = (
        float(df_proj.head(90)["temp_suelo_c"].mean())
        if "temp_suelo_c" in df_proj.columns
        else float(df_proj.head(90)["temp_media_c"].mean())
    )
    evap_acum_proj = (
        float(df_proj.head(120)["evapotranspiracion_mm"].sum())
        if "evapotranspiracion_mm" in df_proj.columns
        else 180.0
    )
    evap_diaria_actual = (
        float(df_hist.tail(30)["evapotranspiracion_mm"].mean())
        if "evapotranspiracion_mm" in df_hist.columns
        else 1.5
    )
    humedad_sup_actual = (
        float(df_hist.tail(30)["humedad_suelo_sup"].mean())
        if "humedad_suelo_sup" in df_hist.columns
        else humedad_raiz_actual
    )
    humedad_prof_actual = (
        float(df_hist.tail(30)["humedad_suelo_prof"].mean())
        if "humedad_suelo_prof" in df_hist.columns
        else humedad_raiz_actual
    )
    humedad_rel_actual = (
        float(df_hist.tail(30)["humedad_relativa_pct"].mean())
        if "humedad_relativa_pct" in df_hist.columns
        else 65.0
    )
    radiacion_actual = (
        float(df_hist.tail(30)["radiacion_solar"].mean())
        if "radiacion_solar" in df_hist.columns
        else 18.0
    )
    temp_media_actual = float(df_hist.tail(30)["temp_media_c"].mean())
    temp_max_actual = float(df_hist.tail(30)["temp_max_c"].mean())
    temp_min_actual = float(df_hist.tail(30)["temp_min_c"].mean())

    # 6. Indicadores espectrales Copernicus Sentinel-2 L2A (NDVI, EVI, NDMI, SUELO_DESNUDO)
    ndvi_actual = (
        float(df_hist.tail(30)["NDVI"].mean())
        if "NDVI" in df_hist.columns
        else 0.52
    )
    ndvi_proj_12m = (
        float(df_proj["NDVI"].mean())
        if "NDVI" in df_proj.columns
        else 0.54
    )
    evi_actual = (
        float(df_hist.tail(30)["EVI"].mean())
        if "EVI" in df_hist.columns
        else 0.38
    )
    ndmi_actual = (
        float(df_hist.tail(30)["NDMI"].mean())
        if "NDMI" in df_hist.columns
        else 0.16
    )
    ndmi_proj_12m = (
        float(df_proj["NDMI"].mean())
        if "NDMI" in df_proj.columns
        else 0.15
    )
    suelo_desnudo_pct = (
        float(ultimos_365["SUELO_DESNUDO"].mean() * 100.0)
        if "SUELO_DESNUDO" in ultimos_365.columns
        else 22.0
    )
    suelo_desnudo_dias_anual = (
        int(ultimos_365["SUELO_DESNUDO"].sum())
        if "SUELO_DESNUDO" in ultimos_365.columns
        else 80
    )

    return {
        "gdd_hist_120": round(float(gdd_hist_120), 1),
        "gdd_proj_120": round(float(gdd_proj_120), 1),
        "dias_helada_anual": dias_helada_anual,
        "dias_riesgo_cultivo_proj": dias_riesgo_cultivo_proj,
        "precip_ultimo_anio_mm": round(float(precip_ultimo_anio), 1),
        "precip_media_anual_mm": round(float(precip_media_anual), 1),
        "anomalia_precip_pct": round(float(anomalia_precip_pct), 1),
        "humedad_sup_actual": round(humedad_sup_actual, 3),
        "humedad_raiz_actual": round(humedad_raiz_actual, 3),
        "humedad_raiz_proj": round(humedad_raiz_proj, 3),
        "delta_humedad_raiz": round(delta_humedad_raiz, 3),
        "humedad_prof_actual": round(humedad_prof_actual, 3),
        "temp_media_actual_c": round(temp_media_actual, 1),
        "temp_max_actual_c": round(temp_max_actual, 1),
        "temp_min_actual_c": round(temp_min_actual, 1),
        "temp_suelo_actual_c": round(temp_suelo_actual, 1),
        "temp_suelo_proj_c": round(temp_suelo_proj, 1),
        "evap_diaria_actual_mm": round(evap_diaria_actual, 2),
        "evap_acum_120d_mm": round(evap_acum_proj, 1),
        "humedad_rel_actual_pct": round(humedad_rel_actual, 1),
        "radiacion_actual": round(radiacion_actual, 1),
        "temp_media_proj_c": round(float(df_proj["temp_media_c"].mean()), 1),
        "ndvi_actual": round(ndvi_actual, 2),
        "ndvi_proj_12m": round(ndvi_proj_12m, 2),
        "evi_actual": round(evi_actual, 2),
        "ndmi_actual": round(ndmi_actual, 2),
        "ndmi_proj_12m": round(ndmi_proj_12m, 2),
        "suelo_desnudo_pct": round(suelo_desnudo_pct, 1),
        "suelo_desnudo_dias_anual": suelo_desnudo_dias_anual,
    }


def _evaluar_aptitud_cultivo(
    fila_cultivo: pd.Series,
    cultivo_actual: pd.Series,
    indicadores: Dict[str, Any],
    df_proj: pd.DataFrame,
    disponibilidad_agua: str,
    modo_estrategia: str,
) -> Tuple[float, Dict[str, Any]]:
    """Calcula un puntaje multicriterio (0-100) y métricas detalladas para un cultivo candidato."""
    puntaje = 55.0

    misma_familia = fila_cultivo["familia_botanica"] == cultivo_actual["familia_botanica"]
    mismo_cultivo = fila_cultivo["cultivo"] == cultivo_actual["cultivo"]

    # Penalización fuerte por repetir familia botánica (corte de ciclo fitosanitario)
    if mismo_cultivo:
        puntaje -= 45.0
    elif misma_familia:
        puntaje -= 30.0
    else:
        puntaje += 15.0

    # Evaluar GDD específicos del cultivo candidato en su ciclo de días
    ciclo = int(fila_cultivo.get("ciclo_dias", 110))
    ventana_proj = df_proj.head(min(ciclo, len(df_proj)))
    gdd_disp = float(np.maximum(0.0, ventana_proj["temp_media_c"] - fila_cultivo["t_base_gdd"]).sum())
    ratio_gdd = gdd_disp / max(1.0, float(fila_cultivo["gdd_requeridos"]))

    if ratio_gdd >= 0.95:
        puntaje += 14.0
    elif ratio_gdd >= 0.75:
        puntaje += 6.0
    else:
        puntaje -= 12.0

    # Evaluar riesgo de heladas según tolerancia térmica del candidato
    dias_helada_cand = int((ventana_proj["temp_min_c"] <= fila_cultivo["t_min_helada"]).sum())
    if dias_helada_cand == 0:
        puntaje += 10.0
    elif dias_helada_cand <= 4:
        puntaje -= 4.0
    else:
        puntaje -= 18.0

    # Ajuste según estrategia
    demanda = fila_cultivo["demanda_hidrica"]
    kc = float(fila_cultivo["kc_promedio"])
    aporte = str(fila_cultivo["aporte_suelo"])

    if modo_estrategia == "ahorro_hidrico":
        if demanda == "Baja":
            puntaje += 22.0
        elif demanda == "Media":
            puntaje += 8.0
        else:
            puntaje -= 20.0
        puntaje += (0.95 - kc) * 35.0
        if disponibilidad_agua == "Baja" and demanda == "Baja":
            puntaje += 8.0

    elif modo_estrategia == "salud_suelo":
        if "Fijador de Nitrógeno" in aporte:
            puntaje += 25.0
        if "Cobertura" in aporte or "Sanitizante" in aporte or "Descompactador" in aporte:
            puntaje += 15.0
        if "Extractor" in str(cultivo_actual["aporte_suelo"]) and "Fijador" in aporte:
            puntaje += 10.0

    elif modo_estrategia == "estabilidad_termica":
        if dias_helada_cand == 0 and ratio_gdd >= 0.9:
            puntaje += 20.0
        if fila_cultivo["t_min_helada"] <= -2.5:
            puntaje += 10.0

    puntaje_final = float(np.clip(round(puntaje, 1), 12.0, 99.0))
    ahorro_agua_pct = round((1.0 - (kc / max(0.5, float(cultivo_actual["kc_promedio"])))) * 100.0, 1)

    detalles = {
        "gdd_disponibles": round(gdd_disp, 0),
        "gdd_requeridos": int(fila_cultivo["gdd_requeridos"]),
        "cumplimiento_gdd_pct": int(min(130, round(ratio_gdd * 100))),
        "dias_riesgo_helada": dias_helada_cand,
        "ahorro_agua_vs_actual_pct": ahorro_agua_pct,
    }
    return puntaje_final, detalles


def construir_secuencia_optima_12m(
    cultivo_actual: pd.Series,
    primer_cultivo_optimo: Dict[str, Any],
    segundo_cultivo_alt: Dict[str, Any],
    df_hort: pd.DataFrame,
    indicadores: Dict[str, Any],
    disponibilidad_agua: str,
    prioridad_usuario: str,
) -> Dict[str, Any]:
    """Construye 1 única secuencia de rotación óptima encadenada para los próximos 12 meses
    (48 bloques semanales: 12 meses x 4 semanas), alternando familias botánicas,
    mejorando el balance hídrico (NDMI/GWETROOT) y reduciendo SUELO_DESNUDO a 0%.
    """
    # Etapa 1: El cultivo ganador según prioridad (ej. Mes 1 a Mes 4/5)
    c1_nombre = primer_cultivo_optimo["cultivo_recomendado"]
    c1_fila = df_hort[df_hort["cultivo"] == c1_nombre].iloc[0]

    # Etapa 2: Siguiente cultivo de distinta familia botánica (ej. Mes 5 a Mes 8/9)
    c2_nombre = segundo_cultivo_alt["cultivo_recomendado"]
    c2_fila = df_hort[df_hort["cultivo"] == c2_nombre].iloc[0]
    if c2_fila["familia_botanica"] == c1_fila["familia_botanica"]:
        alternativas_c2 = df_hort[
            (df_hort["familia_botanica"] != c1_fila["familia_botanica"])
            & (df_hort["familia_botanica"] != cultivo_actual["familia_botanica"])
        ]
        if not alternativas_c2.empty:
            c2_fila = alternativas_c2.iloc[0]
            c2_nombre = c2_fila["cultivo"]

    # Etapa 3: Cierre regenerativo / cobertura o cultivo corto de invierno-primavera (Mes 9 a Mes 12)
    familias_usadas = {
        cultivo_actual["familia_botanica"],
        c1_fila["familia_botanica"],
        c2_fila["familia_botanica"],
    }
    # Priorizar cultivo de cobertura o leguminosa/hortaliza ligera para eliminar SUELO_DESNUDO
    candidatos_c3 = df_hort[~df_hort["cultivo"].isin([c1_nombre, c2_nombre, cultivo_actual["cultivo"]])].copy()
    preferidos_c3 = candidatos_c3[
        candidatos_c3["cultivo"].str.contains("Cobertura|Lechuga|Zanahoria|Haba", case=False, regex=True)
    ]
    if not preferidos_c3.empty:
        # Elegir uno cuya familia no repita la de c2
        distinta_fam_c2 = preferidos_c3[preferidos_c3["familia_botanica"] != c2_fila["familia_botanica"]]
        c3_fila = distinta_fam_c2.iloc[0] if not distinta_fam_c2.empty else preferidos_c3.iloc[0]
    else:
        c3_fila = candidatos_c3.iloc[0]

    # Distribuir las 48 semanas (12 meses x 4 semanas) proporcionalmente a sus ciclos sin dejar suelo desnudo
    s1 = int(np.clip(round(c1_fila["ciclo_dias"] / 7.5), 14, 18))
    s2 = int(np.clip(round(c2_fila["ciclo_dias"] / 7.5), 14, 17))
    s3 = 48 - (s1 + s2)
    if s3 < 12:
        ajuste = 12 - s3
        s1 -= ajuste // 2
        s2 -= ajuste - (ajuste // 2)
        s3 = 48 - (s1 + s2)

    etapas_12m = []
    definiciones = [
        (1, c1_fila, 1, s1, "Ciclo Principal (Meses 1–4)"),
        (2, c2_fila, s1 + 1, s1 + s2, "Ciclo Intermedio (Meses 5–8)"),
        (3, c3_fila, s1 + s2 + 1, 48, "Ciclo Regenerativo / Cobertura (Meses 9–12)"),
    ]

    for num_etapa, fila, sem_ini, sem_fin, etiqueta_periodo in definiciones:
        duracion_sem = sem_fin - sem_ini + 1
        sem_siembra = max(2, int(round(duracion_sem * 0.18)))
        sem_cosecha = max(2, int(round(duracion_sem * 0.20)))
        sem_crecimiento = max(1, duracion_sem - sem_siembra - sem_cosecha)

        plan_48 = []
        for w in range(1, 49):
            if w < sem_ini or w > sem_fin:
                plan_48.append({"activa": False, "fase": "Inactivo"})
            else:
                pos = w - sem_ini
                if pos < sem_siembra:
                    plan_48.append({"activa": True, "fase": "Siembra"})
                elif pos < sem_siembra + sem_crecimiento:
                    plan_48.append({"activa": True, "fase": "Crecimiento"})
                else:
                    plan_48.append({"activa": True, "fase": "Cosecha"})

        etapas_12m.append({
            "orden": num_etapa,
            "etiqueta_periodo": etiqueta_periodo,
            "cultivo": fila["cultivo"],
            "familia_botanica": fila["familia_botanica"],
            "demanda_hidrica": fila["demanda_hidrica"],
            "kc_promedio": float(fila["kc_promedio"]),
            "aporte_suelo": fila["aporte_suelo"],
            "descripcion_beneficio": fila["descripcion_beneficio"],
            "semana_inicio": sem_ini,
            "semana_fin": sem_fin,
            "duracion_semanas": duracion_sem,
            "mes_inicio": ((sem_ini - 1) // 4) + 1,
            "mes_fin": ((sem_fin - 1) // 4) + 1,
            "semanas_48": plan_48,
        })

    kc_medio_secuencia = round(float(np.mean([e["kc_promedio"] for e in etapas_12m])), 2)
    kc_actual = float(cultivo_actual["kc_promedio"])
    ahorro_hidrico_anual_pct = round(max(5.0, (1.0 - (kc_medio_secuencia / max(0.5, kc_actual))) * 100.0 + 12.0), 1)
    cobertura_anual_pct = 100.0  # Las 48 semanas quedan cubiertas sin barbecho desnudo

    resumen_explicativo = (
        f"Esta rotación continua de 12 meses ({c1_fila['cultivo']} ➔ {c2_fila['cultivo']} ➔ {c3_fila['cultivo']}) "
        f"alterna 3 familias botánicas distintas para cortar al 100% las plagas de {cultivo_actual['familia_botanica']}, "
        f"mejora la retención de humedad foliar/radicular (NDMI={indicadores['ndmi_actual']:+.2f}, Kc medio={kc_medio_secuencia}) "
        f"y reduce la exposición de SUELO_DESNUDO del {indicadores['suelo_desnudo_pct']:.0f}% actual a 0% en el año."
    )

    return {
        "nombre_secuencia": f"{c1_fila['cultivo']} ➔ {c2_fila['cultivo']} ➔ {c3_fila['cultivo']}",
        "viabilidad_anual_pct": primer_cultivo_optimo["puntaje_viabilidad"],
        "kc_medio_anual": kc_medio_secuencia,
        "ahorro_hidrico_anual_pct": ahorro_hidrico_anual_pct,
        "suelo_desnudo_actual_pct": indicadores["suelo_desnudo_pct"],
        "suelo_desnudo_meta_pct": 0.0,
        "cobertura_anual_pct": cobertura_anual_pct,
        "resumen_explicativo": resumen_explicativo,
        "etapas": etapas_12m,
    }


def generar_escenarios_rotacion(
    zona: str,
    nombre_cultivo_actual: str,
    disponibilidad_agua: str,
    prioridad_usuario: str,
    df_hist_personalizado: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Ejecuta el motor completo: carga NASA POWER (o el DataFrame interpolado del mapa)
    + proyecta 12 meses + evalúa 3 escenarios y la secuencia anual consolidada.
    """
    df_hist = (
        df_hist_personalizado.copy()
        if df_hist_personalizado is not None
        else cargar_datos_zona(zona)
    )
    df_hort = cargar_catalogo_hortalizas()

    fila_actual_df = df_hort[df_hort["cultivo"] == nombre_cultivo_actual]
    if fila_actual_df.empty:
        cultivo_actual = df_hort.iloc[0]
    else:
        cultivo_actual = fila_actual_df.iloc[0]

    df_proj = proyectar_clima_6_meses(df_hist, dias_horizonte=365)
    indicadores = calcular_indicadores_agronomicos(
        df_hist,
        df_proj,
        t_base_cultivo=float(cultivo_actual["t_base_gdd"]),
        t_min_helada_cultivo=float(cultivo_actual["t_min_helada"]),
    )

    # -------------------------------------------------------------------------
    # ESCENARIO 1: Monocultivo / Rotación Tradicional (Repetir cultivo o familia)
    # -------------------------------------------------------------------------
    puntaje_mono, det_mono = _evaluar_aptitud_cultivo(
        cultivo_actual, cultivo_actual, indicadores, df_proj, disponibilidad_agua, "monocultivo"
    )
    riesgos_mono = [
        f"Repetición de familia {cultivo_actual['familia_botanica']}: multiplica carga de patógenos y nemátodos en el suelo.",
        f"Agotamiento selectivo de nutrientes ({cultivo_actual['aporte_suelo']}) en el mismo horizonte radicular.",
    ]
    if disponibilidad_agua == "Baja" and cultivo_actual["demanda_hidrica"] in ["Media", "Alta"]:
        riesgos_mono.append(
            f"Déficit hídrico crítico: el cultivo exige demanda {cultivo_actual['demanda_hidrica']} (Kc={cultivo_actual['kc_promedio']}) con baja disponibilidad de agua."
        )
    if indicadores["delta_humedad_raiz"] < -0.02:
        riesgos_mono.append(
            f"NASA GWETROOT proyecta caída de humedad radicular ({indicadores['delta_humedad_raiz']:+.2f}) en los próximos 6 meses."
        )

    escenario_1 = {
        "titulo": "Escenario 1: Monocultivo / Tradicional",
        "subtitulo": "Riesgo de mantener el mismo cultivo o familia",
        "cultivo_recomendado": f"Repetir {cultivo_actual['cultivo']}",
        "familia_botanica": cultivo_actual["familia_botanica"],
        "temporada_optima": cultivo_actual["temporada_optima"],
        "demanda_hidrica": cultivo_actual["demanda_hidrica"],
        "kc_promedio": float(cultivo_actual["kc_promedio"]),
        "ciclo_dias": int(cultivo_actual["ciclo_dias"]),
        "aporte_suelo": cultivo_actual["aporte_suelo"],
        "puntaje_viabilidad": min(38.0, puntaje_mono),
        "nivel_riesgo": "ALTO RIESGO",
        "color_badge": "#ef4444",
        "detalles": det_mono,
        "justificacion": (
            f"Repetir {cultivo_actual['cultivo']} ({cultivo_actual['familia_botanica']}) en {zona} "
            "incrementa severamente la presión de enfermedades radiculares, fatiga la fertilidad natural "
            "y reduce el rendimiento esperado entre un 20% y 35%."
        ),
        "puntos_clave": riesgos_mono,
    }

    # -------------------------------------------------------------------------
    # ESCENARIO 2: Estrategia de Ahorro Hídrico y Eficiencia Climática
    # -------------------------------------------------------------------------
    candidatos_distinta_familia = df_hort[
        df_hort["familia_botanica"] != cultivo_actual["familia_botanica"]
    ].copy()

    mejor_hidrico = None
    mejor_puntaje_hid = -1.0
    mejor_det_hid = {}

    for _, fila in candidatos_distinta_familia.iterrows():
        p, det = _evaluar_aptitud_cultivo(
            fila, cultivo_actual, indicadores, df_proj, disponibilidad_agua, "ahorro_hidrico"
        )
        # Evitar elegir cobertura no comercial como primera opción hídrica si hay hortalizas comerciales viables
        if "Cobertura" in fila["cultivo"]:
            p -= 6.0
        if p > mejor_puntaje_hid:
            mejor_puntaje_hid = p
            mejor_hidrico = fila
            mejor_det_hid = det

    escenario_2 = {
        "titulo": "Escenario 2: Estrategia Ahorro Hídrico",
        "subtitulo": "Mínima huella de agua + corte de plagas",
        "cultivo_recomendado": mejor_hidrico["cultivo"],
        "familia_botanica": mejor_hidrico["familia_botanica"],
        "temporada_optima": mejor_hidrico["temporada_optima"],
        "demanda_hidrica": mejor_hidrico["demanda_hidrica"],
        "kc_promedio": float(mejor_hidrico["kc_promedio"]),
        "ciclo_dias": int(mejor_hidrico["ciclo_dias"]),
        "aporte_suelo": mejor_hidrico["aporte_suelo"],
        "puntaje_viabilidad": mejor_puntaje_hid,
        "nivel_riesgo": "ÓPTIMO HÍDRICO",
        "color_badge": "#0ea5e9",
        "detalles": mejor_det_hid,
        "justificacion": mejor_hidrico["descripcion_beneficio"],
        "puntos_clave": [
            f"Cambia de {cultivo_actual['familia_botanica']} a {mejor_hidrico['familia_botanica']}, rompiendo el ciclo biológico de plagas.",
            f"Coeficiente de cultivo Kc={mejor_hidrico['kc_promedio']} ({mejor_det_hid['ahorro_agua_vs_actual_pct']:+.1f}% de variación en demanda evaporativa frente a {cultivo_actual['cultivo']}).",
            f"Adecuado para humedad radicular NASA proyectada en {zona} ({indicadores['humedad_raiz_proj']*100:.0f}% de saturación).",
        ],
    }

    # -------------------------------------------------------------------------
    # ESCENARIO 3: Estrategia Regenerativa de Salud de Suelo y Nitrógeno
    # -------------------------------------------------------------------------
    mejor_suelo = None
    mejor_puntaje_suelo = -1.0
    mejor_det_suelo = {}

    for _, fila in candidatos_distinta_familia.iterrows():
        # Evitar duplicar exactamente el mismo cultivo del escenario 2 para ofrecer 2 alternativas ricas al agricultor
        if fila["cultivo"] == mejor_hidrico["cultivo"]:
            continue
        p, det = _evaluar_aptitud_cultivo(
            fila, cultivo_actual, indicadores, df_proj, disponibilidad_agua, "salud_suelo"
        )
        if p > mejor_puntaje_suelo:
            mejor_puntaje_suelo = p
            mejor_suelo = fila
            mejor_det_suelo = det

    escenario_3 = {
        "titulo": "Escenario 3: Salud de Suelo y Bio-regeneración",
        "subtitulo": "Fijación de nitrógeno + supresión fitosanitaria",
        "cultivo_recomendado": mejor_suelo["cultivo"],
        "familia_botanica": mejor_suelo["familia_botanica"],
        "temporada_optima": mejor_suelo["temporada_optima"],
        "demanda_hidrica": mejor_suelo["demanda_hidrica"],
        "kc_promedio": float(mejor_suelo["kc_promedio"]),
        "ciclo_dias": int(mejor_suelo["ciclo_dias"]),
        "aporte_suelo": mejor_suelo["aporte_suelo"],
        "puntaje_viabilidad": mejor_puntaje_suelo,
        "nivel_riesgo": "REGENERATIVO",
        "color_badge": "#10b981",
        "detalles": mejor_det_suelo,
        "justificacion": mejor_suelo["descripcion_beneficio"],
        "puntos_clave": [
            f"Aporte directo al suelo: {mejor_suelo['aporte_suelo']} tras un cultivo {cultivo_actual['aporte_suelo'].lower()}.",
            f"Temperatura de suelo proyectada en {zona} ({indicadores['temp_suelo_proj_c']} °C) favorece la actividad microbiana y radicular.",
            f"Cumplimiento térmico proyectado del {mejor_det_suelo['cumplimiento_gdd_pct']}% de los GDD requeridos ({mejor_det_suelo['gdd_disponibles']:.0f}/{mejor_det_suelo['gdd_requeridos']} GDD).",
        ],
    }

    # Determinar cuál escenario destacar según la prioridad elegida por el usuario
    if "agua" in prioridad_usuario.lower() or disponibilidad_agua == "Baja":
        recomendacion_principal = escenario_2
        estrategia_ganadora = "Escenario 2 (Ahorro Hídrico)"
    elif "suelo" in prioridad_usuario.lower() or "nitrógeno" in prioridad_usuario.lower():
        recomendacion_principal = escenario_3
        estrategia_ganadora = "Escenario 3 (Salud de Suelo)"
    else:
        # Prioridad: Estabilidad térmica -> elegir entre escenario 2 y 3 el de menor riesgo de helada y mejor cumplimiento GDD
        if (
            escenario_2["detalles"]["dias_riesgo_helada"]
            <= escenario_3["detalles"]["dias_riesgo_helada"]
        ):
            recomendacion_principal = escenario_2
            estrategia_ganadora = "Escenario 2 (Óptimo Térmico-Hídrico)"
        else:
            recomendacion_principal = escenario_3
            estrategia_ganadora = "Escenario 3 (Óptimo Térmico-Regenerativo)"

    # Generar alertas agronómicas inteligentes basadas en datos reales de NASA POWER
    alertas = []
    if indicadores["anomalia_precip_pct"] < -15.0:
        alertas.append({
            "tipo": "warning",
            "titulo": "Anomalía Pluviométrica Negativa Detectada",
            "mensaje": (
                f"En {zona}, la precipitación del último año ({indicadores['precip_ultimo_anio_mm']} mm) "
                f"está un {abs(indicadores['anomalia_precip_pct']):.1f}% por debajo del promedio de la serie. "
                "Se recomienda priorizar cultivos con Kc <= 0.80 y cobertura vegetal."
            ),
        })
    elif indicadores["anomalia_precip_pct"] > 15.0:
        alertas.append({
            "tipo": "info",
            "titulo": "Superávit Hídrico Reciente",
            "mensaje": (
                f"La precipitación del último año ({indicadores['precip_ultimo_anio_mm']} mm) supera en "
                f"{indicadores['anomalia_precip_pct']:.1f}% la media multianual. Vigilar drenaje y hongos radiculares en Solanáceas."
            ),
        })

    if indicadores["dias_riesgo_cultivo_proj"] > 0:
        alertas.append({
            "tipo": "warning",
            "titulo": f"Alerta Térmica: {indicadores['dias_riesgo_cultivo_proj']} días con riesgo de helada proyectados",
            "mensaje": (
                f"El modelo proyecta temperaturas mínimas bajo el umbral crítico de {cultivo_actual['cultivo']} "
                f"({cultivo_actual['t_min_helada']} °C). Rotar hacia especies tolerantes al frío ({recomendacion_principal['cultivo_recomendado']})."
            ),
        })
    else:
        alertas.append({
            "tipo": "success",
            "titulo": "Ventana Térmica Favorable (Próximos 6 Meses)",
            "mensaje": (
                f"Acumulación térmica proyectada de {indicadores['gdd_proj_120']} GDD (base {cultivo_actual['t_base_gdd']} °C) "
                f"y temperatura media de suelo de {indicadores['temp_suelo_proj_c']} °C sin heladas críticas."
            ),
        })

    # Construir la Secuencia Óptima Consolidada de 12 Meses (48 semanas) para la Carta Gantt
    segundo_cultivo_sec = escenario_3 if recomendacion_principal == escenario_2 else escenario_2
    secuencia_optima_12m = construir_secuencia_optima_12m(
        cultivo_actual=cultivo_actual,
        primer_cultivo_optimo=recomendacion_principal,
        segundo_cultivo_alt=segundo_cultivo_sec,
        df_hort=df_hort,
        indicadores=indicadores,
        disponibilidad_agua=disponibilidad_agua,
        prioridad_usuario=prioridad_usuario,
    )

    return {
        "zona": zona,
        "cultivo_actual": cultivo_actual.to_dict(),
        "indicadores": indicadores,
        "df_historico": df_hist,
        "df_proyeccion": df_proj,
        "escenarios": [escenario_1, escenario_2, escenario_3],
        "recomendacion_principal": recomendacion_principal,
        "estrategia_ganadora": estrategia_ganadora,
        "secuencia_optima_12m": secuencia_optima_12m,
        "alertas": alertas,
    }


if __name__ == "__main__":
    resultado = generar_escenarios_rotacion(
        zona="Quillota",
        nombre_cultivo_actual="Tomate",
        disponibilidad_agua="Baja",
        prioridad_usuario="Ahorrar agua",
    )
    print(f"Zona: {resultado['zona']} | Ganador: {resultado['estrategia_ganadora']}")
    print("Indicadores NASA:", resultado["indicadores"])
    for esc in resultado["escenarios"]:
        print(
            f" -> {esc['titulo']}: {esc['cultivo_recomendado']} "
            f"(Viabilidad: {esc['puntaje_viabilidad']}% | Kc: {esc['kc_promedio']})"
        )
