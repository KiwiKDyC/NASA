from pathlib import Path
from typing import Dict, Any, Tuple, List
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
    e índices espectrales Sentinel-2 (NDVI, EVI, NDMI, SUELO_DESNUDO) para el horizonte indicado
    (365 o 730 días) usando regresión lineal armónica (tendencia + estacionalidad de Fourier).
    """
    df = df_hist.copy()
    fecha_min = df["fecha"].min()
    t = (df["fecha"] - fecha_min).dt.days.to_numpy(dtype=float)
    doy = df["fecha"].dt.dayofyear.to_numpy(dtype=float)

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

    df["doy"] = df["fecha"].dt.dayofyear
    precip_clim = df.groupby("doy")["precipitacion_mm"].mean().to_dict()
    df_proj["precipitacion_mm"] = [
        round(precip_clim.get(d, 0.5), 2) for d in fechas_futuras.dayofyear
    ]

    # Calcular Evapotranspiración de Referencia (ET_ref en mm/día) combinando NASA EVLAND y radiación/temperatura
    # En climas mediterráneos semiáridos (Chile Central), EVLAND mide evaporación real de suelo y Hargreaves/FAO
    # con radiación solar (ALLSKY_SFC_SW_DWN) y temperatura (T2M) da la demanda evaporativa de referencia (ET_ref)
    rad = df_proj["radiacion_solar"].to_numpy(dtype=float) if "radiacion_solar" in df_proj.columns else np.full(len(df_proj), 18.0)
    t_med = df_proj["temp_media_c"].to_numpy(dtype=float)
    evland = df_proj["evapotranspiracion_mm"].to_numpy(dtype=float) if "evapotranspiracion_mm" in df_proj.columns else np.full(len(df_proj), 2.2)
    # Ecuación FAO-Hargreaves calibrada con NASA ALLSKY_SFC_SW_DWN y T2M + EVLAND (rango Valle Central: 1.8 a 6.5 mm/día)
    et_ref_fao = np.clip(0.0135 * (t_med + 17.78) * (rad / 2.45) * 0.88 + 0.45 * evland, 1.8, 6.8)
    df_proj["et_ref_mm"] = np.round(et_ref_fao, 2)

    df_proj["tipo_dato"] = "Proyección Satelital (IA/Estacional)"
    return df_proj


proyectar_clima_12_meses = proyectar_clima_6_meses


def calcular_ciclo_dinamico_gdd(
    fila_cultivo: pd.Series,
    df_proj: pd.DataFrame,
    dia_inicio: int = 0,
) -> Dict[str, Any]:
    """Determina dinámicamente la duración del ciclo del cultivo (en días y semanas)
    según la temperatura proyectada de la zona evaluada y los Grados-Día (GDD) requeridos:
    - En zonas o estaciones más cálidas, los GDD se acumulan más rápido -> el ciclo se acorta.
    - En zonas más frías o en invierno, la acumulación térmica diaria es menor -> el ciclo se alarga.
    """
    ciclo_nominal = int(fila_cultivo.get("ciclo_dias", 100))
    t_base = float(fila_cultivo.get("t_base_gdd", 6.0))
    gdd_req = float(fila_cultivo.get("gdd_requeridos", 1000.0))
    es_puente = bool(fila_cultivo.get("es_puente_cobertura", False))

    min_dias = int(round(ciclo_nominal * (0.80 if es_puente else 0.78)))
    max_dias = int(round(ciclo_nominal * (1.22 if es_puente else 1.32)))

    sub_df = df_proj.iloc[dia_inicio : min(len(df_proj), dia_inicio + max_dias + 30)]
    if sub_df.empty:
        sub_df = df_proj.head(max_dias)

    gdd_diarios = np.maximum(0.5, sub_df["temp_media_c"].to_numpy(dtype=float) - t_base)
    gdd_acum = np.cumsum(gdd_diarios)

    # Buscar el día exacto en que alcanza los GDD requeridos
    indices_cumple = np.where(gdd_acum >= gdd_req)[0]
    if len(indices_cumple) > 0:
        dias_gdd_puro = int(indices_cumple[0]) + 1
    else:
        tasa_media = float(np.mean(gdd_diarios)) if len(gdd_diarios) > 0 else 8.0
        dias_gdd_puro = int(round(gdd_req / max(2.5, tasa_media)))

    ciclo_dias_dinamico = int(np.clip(dias_gdd_puro, min_dias, max_dias))
    ventana_real = sub_df.head(ciclo_dias_dinamico)
    gdd_logrados = float(
        np.maximum(0.0, ventana_real["temp_media_c"].to_numpy(dtype=float) - t_base).sum()
    )
    cumplimiento_pct = int(min(130, max(78, round((gdd_logrados / max(1.0, gdd_req)) * 100))))

    semanas_dinamicas = max(8 if es_puente else 11, int(round(ciclo_dias_dinamico / 7.0)))
    variacion_dias = ciclo_dias_dinamico - ciclo_nominal

    return {
        "ciclo_nominal_dias": ciclo_nominal,
        "ciclo_dinamico_dias": ciclo_dias_dinamico,
        "variacion_dias_vs_nominal": variacion_dias,
        "semanas_dinamicas": semanas_dinamicas,
        "gdd_requeridos": int(round(gdd_req)),
        "gdd_acumulados_ciclo": round(gdd_logrados, 0),
        "cumplimiento_gdd_pct": cumplimiento_pct,
    }


def calcular_et_cultivo_nasa(
    df_proj: pd.DataFrame,
    kc_cultivo: float,
    dia_inicio: int,
    duracion_dias: int,
) -> Dict[str, Any]:
    """Calcula la Evapotranspiración del Cultivo (ET_c = Kc * ET_ref) extrayendo la
    Evapotranspiración de Referencia (ET_ref) de los datos NASA POWER en los días exactos
    del ciclo del cultivo, y la convierte a consumo de agua en m³/ha (1 mm = 10 m³/ha).
    """
    sub_df = df_proj.iloc[dia_inicio : min(len(df_proj), dia_inicio + duracion_dias)]
    if sub_df.empty:
        sub_df = df_proj.head(duracion_dias)

    if "et_ref_mm" in sub_df.columns:
        et_ref_serie = sub_df["et_ref_mm"].to_numpy(dtype=float)
    else:
        et_ref_serie = sub_df["evapotranspiracion_mm"].to_numpy(dtype=float) * 1.65

    factor_escala = duracion_dias / max(1, len(et_ref_serie))
    et_ref_total_mm = float(np.sum(et_ref_serie) * factor_escala)
    et_ref_diaria_mm = float(np.mean(et_ref_serie))

    # Ecuación agronómica FAO / NASA: ET_c = Kc * ET_ref
    etc_total_mm = et_ref_total_mm * float(kc_cultivo)
    etc_diaria_mm = et_ref_diaria_mm * float(kc_cultivo)
    consumo_agua_m3_ha = int(round(etc_total_mm * 10.0))  # 1 mm de lámina = 10 m3/ha

    return {
        "et_ref_total_mm": round(et_ref_total_mm, 1),
        "et_ref_diaria_mm": round(et_ref_diaria_mm, 2),
        "etc_total_mm": round(etc_total_mm, 1),
        "etc_diaria_mm": round(etc_diaria_mm, 2),
        "consumo_agua_m3_ha": consumo_agua_m3_ha,
    }


def calcular_indicadores_agronomicos(
    df_hist: pd.DataFrame,
    df_proj: pd.DataFrame,
    t_base_cultivo: float = 8.0,
    t_min_helada_cultivo: float = 0.0,
) -> Dict[str, Any]:
    """Calcula indicadores agroclimáticos históricos y proyectados a partir de NASA POWER."""
    ultimos_365 = df_hist.tail(365).copy()

    ventana_120_hist = df_hist.tail(120)
    gdd_hist_120 = np.maximum(0.0, ventana_120_hist["temp_media_c"] - t_base_cultivo).sum()

    ventana_120_proj = df_proj.head(120)
    gdd_proj_120 = np.maximum(0.0, ventana_120_proj["temp_media_c"] - t_base_cultivo).sum()

    dias_helada_anual = int((ultimos_365["temp_min_c"] <= 0.0).sum())
    dias_riesgo_cultivo_proj = int((df_proj["temp_min_c"] <= t_min_helada_cultivo).sum())

    precip_ultimo_anio = ultimos_365["precipitacion_mm"].sum()
    anios_totales = max(1.0, len(df_hist) / 365.25)
    precip_media_anual = df_hist["precipitacion_mm"].sum() / anios_totales
    anomalia_precip_pct = (
        ((precip_ultimo_anio - precip_media_anual) / precip_media_anual) * 100.0
        if precip_media_anual > 0
        else 0.0
    )

    humedad_raiz_actual = float(df_hist.tail(30)["humedad_suelo_raiz"].mean())
    humedad_raiz_proj = float(df_proj["humedad_suelo_raiz"].mean())
    delta_humedad_raiz = humedad_raiz_proj - humedad_raiz_actual

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
    et_ref_diaria_proj = (
        float(df_proj.head(120)["et_ref_mm"].mean())
        if "et_ref_mm" in df_proj.columns
        else round(evap_diaria_actual * 1.6, 2)
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
        "et_ref_diaria_proj_mm": round(et_ref_diaria_proj, 2),
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
    dia_inicio: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Calcula un puntaje multicriterio (0-100) y métricas dinámicas basadas en GDD y ET = Kc*ET_ref."""
    puntaje = 55.0

    misma_familia = fila_cultivo["familia_botanica"] == cultivo_actual["familia_botanica"]
    mismo_cultivo = fila_cultivo["cultivo"] == cultivo_actual["cultivo"]

    if mismo_cultivo:
        puntaje -= 45.0
    elif misma_familia:
        puntaje -= 30.0
    else:
        puntaje += 15.0

    info_gdd = calcular_ciclo_dinamico_gdd(fila_cultivo, df_proj, dia_inicio=dia_inicio)
    ciclo_din = info_gdd["ciclo_dinamico_dias"]
    gdd_disp = info_gdd["gdd_acumulados_ciclo"]
    ratio_gdd = gdd_disp / max(1.0, float(fila_cultivo["gdd_requeridos"]))

    if ratio_gdd >= 0.95:
        puntaje += 14.0
    elif ratio_gdd >= 0.78:
        puntaje += 6.0
    else:
        puntaje -= 12.0

    ventana_proj = df_proj.iloc[dia_inicio : min(len(df_proj), dia_inicio + ciclo_din)]
    if ventana_proj.empty:
        ventana_proj = df_proj.head(ciclo_din)

    dias_helada_cand = int((ventana_proj["temp_min_c"] <= fila_cultivo["t_min_helada"]).sum())
    if dias_helada_cand == 0:
        puntaje += 10.0
    elif dias_helada_cand <= 4:
        puntaje -= 4.0
    else:
        puntaje -= 18.0

    demanda = fila_cultivo["demanda_hidrica"]
    kc = float(fila_cultivo["kc_promedio"])
    aporte = str(fila_cultivo["aporte_suelo"])
    bal_n = int(fila_cultivo.get("balance_nitrogeno_kg_ha", 0))

    info_et = calcular_et_cultivo_nasa(df_proj, kc, dia_inicio=dia_inicio, duracion_dias=ciclo_din)

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
        if bal_n > 0 or "Fijador" in aporte:
            puntaje += 26.0
        if "Cobertura" in aporte or "Sanitizante" in aporte or "Descompactador" in aporte or "Biofumigante" in aporte:
            puntaje += 15.0
        if int(cultivo_actual.get("balance_nitrogeno_kg_ha", -80)) < 0 and bal_n > 0:
            puntaje += 10.0

    elif modo_estrategia == "estabilidad_termica":
        if dias_helada_cand == 0 and ratio_gdd >= 0.9:
            puntaje += 20.0
        if fila_cultivo["t_min_helada"] <= -2.5:
            puntaje += 10.0

    elif modo_estrategia == "super_plan_integral":
        # Combina las 3 variables simultáneamente: Ahorro de Agua + Recuperación de Suelo/N + Resistencia Térmica
        if demanda in ["Baja", "Media"]:
            puntaje += 12.0
        puntaje += (0.92 - kc) * 25.0
        if bal_n > 0 or "Fijador" in aporte or "Sanitizante" in aporte or "Biofumigante" in aporte:
            puntaje += 16.0
        if dias_helada_cand == 0 and ratio_gdd >= 0.88:
            puntaje += 12.0

    puntaje_final = float(np.clip(round(puntaje, 1), 12.0, 99.0))
    ahorro_agua_pct = round((1.0 - (kc / max(0.5, float(cultivo_actual["kc_promedio"])))) * 100.0, 1)

    detalles = {
        "gdd_disponibles": round(gdd_disp, 0),
        "gdd_requeridos": int(fila_cultivo["gdd_requeridos"]),
        "cumplimiento_gdd_pct": info_gdd["cumplimiento_gdd_pct"],
        "ciclo_nominal_dias": info_gdd["ciclo_nominal_dias"],
        "ciclo_dinamico_dias": ciclo_din,
        "variacion_dias_gdd": info_gdd["variacion_dias_vs_nominal"],
        "semanas_dinamicas": info_gdd["semanas_dinamicas"],
        "dias_riesgo_helada": dias_helada_cand,
        "ahorro_agua_vs_actual_pct": ahorro_agua_pct,
        "et_ref_mm": info_et["et_ref_total_mm"],
        "etc_mm": info_et["etc_total_mm"],
        "etc_diaria_mm": info_et["etc_diaria_mm"],
        "consumo_agua_m3_ha": info_et["consumo_agua_m3_ha"],
    }
    return puntaje_final, detalles


def construir_secuencia_optima_12m(
    cultivo_actual: pd.Series,
    primer_cultivo_optimo: Dict[str, Any],
    segundo_cultivo_alt: Dict[str, Any],
    df_hort: pd.DataFrame,
    df_proj: pd.DataFrame,
    indicadores: Dict[str, Any],
    disponibilidad_agua: str,
    prioridad_usuario: str,
) -> Dict[str, Any]:
    """Construye la secuencia de rotación de 12 meses (48 semanas) con duración dinámica por GDD
    y consumo hídrico calculado mediante ET_c = Kc * ET_ref.
    """
    c1_nombre = primer_cultivo_optimo["cultivo_recomendado"]
    c1_fila = df_hort[df_hort["cultivo"] == c1_nombre].iloc[0]

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

    candidatos_c3 = df_hort[~df_hort["cultivo"].isin([c1_nombre, c2_nombre, cultivo_actual["cultivo"]])].copy()
    preferidos_c3 = candidatos_c3[
        candidatos_c3["cultivo"].str.contains("Cobertura|Lechuga|Zanahoria|Haba", case=False, regex=True)
    ]
    if not preferidos_c3.empty:
        distinta_fam_c2 = preferidos_c3[preferidos_c3["familia_botanica"] != c2_fila["familia_botanica"]]
        c3_fila = distinta_fam_c2.iloc[0] if not distinta_fam_c2.empty else preferidos_c3.iloc[0]
    else:
        c3_fila = candidatos_c3.iloc[0]

    gdd1 = calcular_ciclo_dinamico_gdd(c1_fila, df_proj, dia_inicio=0)
    gdd2 = calcular_ciclo_dinamico_gdd(c2_fila, df_proj, dia_inicio=gdd1["ciclo_dinamico_dias"])
    gdd3 = calcular_ciclo_dinamico_gdd(
        c3_fila, df_proj, dia_inicio=gdd1["ciclo_dinamico_dias"] + gdd2["ciclo_dinamico_dias"]
    )

    total_sem_raw = gdd1["semanas_dinamicas"] + gdd2["semanas_dinamicas"] + gdd3["semanas_dinamicas"]
    s1 = max(12, int(round(48 * (gdd1["semanas_dinamicas"] / max(1, total_sem_raw)))))
    s2 = max(12, int(round(48 * (gdd2["semanas_dinamicas"] / max(1, total_sem_raw)))))
    s3 = 48 - (s1 + s2)
    if s3 < 10:
        ajuste = 10 - s3
        s1 -= ajuste // 2
        s2 -= ajuste - (ajuste // 2)
        s3 = 48 - (s1 + s2)

    etapas_12m = []
    definiciones = [
        (1, c1_fila, 1, s1, "Ciclo Principal (Meses 1–4)", gdd1, 0),
        (2, c2_fila, s1 + 1, s1 + s2, "Ciclo Intermedio (Meses 5–8)", gdd2, s1 * 7),
        (3, c3_fila, s1 + s2 + 1, 48, "Ciclo Regenerativo / Cobertura (Meses 9–12)", gdd3, (s1 + s2) * 7),
    ]

    for num_etapa, fila, sem_ini, sem_fin, etiqueta_periodo, info_g, dia_ini in definiciones:
        duracion_sem = sem_fin - sem_ini + 1
        duracion_dias = info_g["ciclo_dinamico_dias"]
        info_et = calcular_et_cultivo_nasa(
            df_proj, float(fila["kc_promedio"]), dia_inicio=dia_ini, duracion_dias=duracion_dias
        )

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
            "ciclo_dinamico_dias": duracion_dias,
            "mes_inicio": ((sem_ini - 1) // 4) + 1,
            "mes_fin": ((sem_fin - 1) // 4) + 1,
            "balance_nitrogeno_kg_ha": int(fila.get("balance_nitrogeno_kg_ha", 0)),
            "et_ref_mm": info_et["et_ref_total_mm"],
            "etc_mm": info_et["etc_total_mm"],
            "consumo_agua_m3_ha": info_et["consumo_agua_m3_ha"],
            "es_puente_cobertura": bool(fila.get("es_puente_cobertura", False)),
            "semanas_48": plan_48,
        })

    kc_medio_secuencia = round(float(np.mean([e["kc_promedio"] for e in etapas_12m])), 2)
    kc_actual = float(cultivo_actual["kc_promedio"])
    ahorro_hidrico_anual_pct = round(max(5.0, (1.0 - (kc_medio_secuencia / max(0.5, kc_actual))) * 100.0 + 12.0), 1)
    cobertura_anual_pct = 100.0

    resumen_explicativo = (
        f"Esta rotación continua ({c1_fila['cultivo']} ➔ {c2_fila['cultivo']} ➔ {c3_fila['cultivo']}) "
        f"alterna familias botánicas distintas para cortar al 100% las plagas de {cultivo_actual['familia_botanica']}, "
        f"ajusta los ciclos según los Grados-Día (GDD) locales de la zona y reduce la exposición de suelo desnudo a 0%."
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


def _repartir_semanas_anio_gdd(
    filas_anio: List[Tuple[pd.Series, bool]],
    df_proj_24m: pd.DataFrame,
    dia_offset_anio: int,
) -> List[Dict[str, Any]]:
    """Calcula la duración en días y semanas de los 3 cultivos de un año (1 Puente + 2 Temporadas)
    usando los Grados-Día (GDD) reales proyectados en la zona para que la Carta Gantt varíe
    dinámicamente según la temperatura local, sumando exactamente 48 semanas por año.
    """
    infos_gdd = []
    dia_cursor = dia_offset_anio
    for fila, es_puente in filas_anio:
        info = calcular_ciclo_dinamico_gdd(fila, df_proj_24m, dia_inicio=dia_cursor)
        infos_gdd.append(info)
        dia_cursor += info["ciclo_dinamico_dias"]

    sem_raw = [inf["semanas_dinamicas"] for inf in infos_gdd]
    # El cultivo puente dura entre 8 y 11 semanas (56 a 77 días según GDD); las temporadas comerciales el resto
    s_puente = int(np.clip(sem_raw[0], 8, 11))
    rem = 48 - s_puente
    prop_t1 = sem_raw[1] / max(1, sem_raw[1] + sem_raw[2])
    s_t1 = int(np.clip(round(rem * prop_t1), 15, rem - 15))
    s_t2 = rem - s_t1

    semanas_asignadas = [s_puente, s_t1, s_t2]
    for idx, s_asig in enumerate(semanas_asignadas):
        infos_gdd[idx]["semanas_gantt"] = s_asig
    return infos_gdd


def construir_plan_24m_y_cuenta_corriente(
    cultivo_actual: pd.Series,
    primer_cultivo_optimo: Dict[str, Any],
    segundo_cultivo_alt: Dict[str, Any],
    df_hort: pd.DataFrame,
    df_proj_24m: pd.DataFrame,
    indicadores: Dict[str, Any],
    disponibilidad_agua: str,
    prioridad_usuario: str,
    pendiente_srtm_pct: float = 4.2,
) -> Dict[str, Any]:
    """Construye el Planificador Multi-Temporada a 2 Años (24 Meses / 96 Semanas):
    1. Determina los ciclos de cada cultivo dinámicamente según los Grados-Día (GDD) de la zona.
    2. Calcula la Evapotranspiración de cada cultivo mediante ET_c = Kc * ET_ref (NASA POWER).
    3. Inserta Abonos Verdes de Ventana Corta (Mostaza Blanca y Vicia) según riesgo de erosión SRTM.
    4. Valida la Cuenta Corriente de Nitrógeno (kg N/ha -> kg Urea 46% N -> $ CLP/ha) y Agua (m³/ha).
    """
    comerciales = df_hort[~df_hort["es_puente_cobertura"]].copy()
    puentes = df_hort[df_hort["es_puente_cobertura"]].copy()

    ventana_70d = df_proj_24m.head(70)
    gdd_ventana_70d = float(np.maximum(0.0, ventana_70d["temp_media_c"] - 4.5).sum())
    lluvia_oto_inv_mm = float(df_proj_24m.head(180)["precipitacion_mm"].sum())
    suelo_desnudo_pct = float(indicadores.get("suelo_desnudo_pct", 24.0))

    indice_erosion = round(
        min(99.0, pendiente_srtm_pct * 4.5 + (lluvia_oto_inv_mm / 450.0) * 30.0 + suelo_desnudo_pct * 1.1),
        1,
    )
    riesgo_erosion_txt = "Alto" if indice_erosion >= 58.0 else ("Medio-Alto" if indice_erosion >= 36.0 else "Moderado")

    previa_es_solanacea = "Solanaceae" in str(cultivo_actual["familia_botanica"])
    balance_n_previo = int(cultivo_actual.get("balance_nitrogeno_kg_ha", -80))

    if previa_es_solanacea:
        fila_puente_1 = puentes[puentes["cultivo"].str.contains("Mostaza", case=False, regex=False)].iloc[0]
        razon_puente_1_corta = (
            f"Como vienes de {cultivo_actual['cultivo']} ({cultivo_actual['familia_botanica']}), la tierra queda "
            f"con riesgo de hongos y nemátodos. Sembrar Mostaza Blanca limpia y desinfecta las raíces de forma natural."
        )
        razon_puente_2_corta = (
            "En el segundo año se siembra Vicia durante la ventana corta para inyectar +90 kg N/ha de nitrógeno "
            "biológico gratuito y proteger el suelo de las lluvias."
        )
    else:
        fila_puente_1 = puentes[puentes["cultivo"].str.contains("Vicia (Abono", case=False, regex=False)].iloc[0]
        razon_puente_1_corta = (
            f"Tras cosechar {cultivo_actual['cultivo']} (que extrae {balance_n_previo} kg N/ha), la Vicia cubre la "
            "tierra contra la erosión e inyecta +90 kg de Nitrógeno natural por hectárea."
        )
        razon_puente_2_corta = (
            "En el segundo año la Mostaza Blanca actúa como biofumigante natural, cortando plagas del suelo y "
            "aportando +30 kg N/ha adicionales."
        )

    # Selección de las 4 temporadas comerciales encadenadas
    t1_nombre = primer_cultivo_optimo["cultivo_recomendado"]
    t1_fila = df_hort[df_hort["cultivo"] == t1_nombre].iloc[0]
    if t1_fila["familia_botanica"] == fila_puente_1["familia_botanica"] or bool(t1_fila.get("es_puente_cobertura", False)):
        cand_t1 = comerciales[
            (comerciales["familia_botanica"] != fila_puente_1["familia_botanica"])
            & (comerciales["familia_botanica"] != cultivo_actual["familia_botanica"])
        ]
        pref_fab = cand_t1[cand_t1["familia_botanica"].str.contains("Fabaceae", case=False)]
        t1_fila = pref_fab.iloc[0] if not pref_fab.empty else cand_t1.iloc[0]

    cand_t2 = comerciales[
        (comerciales["familia_botanica"] != t1_fila["familia_botanica"])
        & (~comerciales["cultivo"].isin([t1_fila["cultivo"], cultivo_actual["cultivo"]]))
    ]
    if not any("Fabaceae" in f for f in [str(fila_puente_1["familia_botanica"]), str(t1_fila["familia_botanica"])]):
        pref_t2 = cand_t2[cand_t2["familia_botanica"].str.contains("Fabaceae", case=False)]
    else:
        pref_t2 = cand_t2[cand_t2["familia_botanica"].str.contains("Brassicaceae|Amaryllidaceae|Poaceae", case=False)]
    t2_fila = pref_t2.iloc[0] if not pref_t2.empty else cand_t2.iloc[0]

    if "Mostaza" in fila_puente_1["cultivo"]:
        fila_puente_2 = puentes[puentes["cultivo"].str.contains("Vicia (Abono", case=False, regex=False)].iloc[0]
    else:
        fila_puente_2 = puentes[puentes["cultivo"].str.contains("Mostaza", case=False, regex=False)].iloc[0]

    cand_t3 = comerciales[
        (comerciales["familia_botanica"] != fila_puente_2["familia_botanica"])
        & (comerciales["familia_botanica"] != t2_fila["familia_botanica"])
        & (~comerciales["cultivo"].isin([t1_fila["cultivo"], t2_fila["cultivo"]]))
    ]
    pref_t3 = cand_t3[cand_t3["familia_botanica"].str.contains("Amaryllidaceae|Brassicaceae|Apiaceae|Asteraceae", case=False)]
    t3_fila = pref_t3.iloc[0] if not pref_t3.empty else cand_t3.iloc[0]

    cand_t4 = comerciales[
        (comerciales["familia_botanica"] != t3_fila["familia_botanica"])
        & (comerciales["familia_botanica"] != cultivo_actual["familia_botanica"])
        & (~comerciales["cultivo"].isin([t1_fila["cultivo"], t2_fila["cultivo"], t3_fila["cultivo"]]))
    ]
    pref_t4 = cand_t4[cand_t4["familia_botanica"].str.contains("Fabaceae|Poaceae|Asteraceae|Apiaceae", case=False)]
    t4_fila = pref_t4.iloc[0] if not pref_t4.empty else cand_t4.iloc[0]

    # --- DURACIÓN DINÁMICA DE CICLOS POR GRADOS-DÍA (GDD) EN CADA AÑO ---
    gdd_anio1 = _repartir_semanas_anio_gdd(
        [(fila_puente_1, True), (t1_fila, False), (t2_fila, False)],
        df_proj_24m,
        dia_offset_anio=0,
    )
    gdd_anio2 = _repartir_semanas_anio_gdd(
        [(fila_puente_2, True), (t3_fila, False), (t4_fila, False)],
        df_proj_24m,
        dia_offset_anio=365,
    )

    s1_a1, s2_a1, s3_a1 = [x["semanas_gantt"] for x in gdd_anio1]
    s1_a2, s2_a2, s3_a2 = [x["semanas_gantt"] for x in gdd_anio2]

    bloques_24m = [
        (1, fila_puente_1, 1, s1_a1, True, "Abono Verde Puente (Año 1)", gdd_anio1[0]),
        (2, t1_fila, s1_a1 + 1, s1_a1 + s2_a1, False, "Temporada Comercial 1 (Año 1)", gdd_anio1[1]),
        (3, t2_fila, s1_a1 + s2_a1 + 1, 48, False, "Temporada Comercial 2 (Año 1)", gdd_anio1[2]),
        (4, fila_puente_2, 49, 48 + s1_a2, True, "Abono Verde Puente (Año 2)", gdd_anio2[0]),
        (5, t3_fila, 48 + s1_a2 + 1, 48 + s1_a2 + s2_a2, False, "Temporada Comercial 3 (Año 2)", gdd_anio2[1]),
        (6, t4_fila, 48 + s1_a2 + s2_a2 + 1, 96, False, "Temporada Comercial 4 (Año 2)", gdd_anio2[2]),
    ]

    etapas_24m = []
    saldo_n_acumulado = 0
    movimientos_cuenta = []

    for orden, fila, sem_ini, sem_fin, es_puente, rol_etapa, info_gdd in bloques_24m:
        duracion_sem = sem_fin - sem_ini + 1
        ciclo_dias_din = info_gdd["ciclo_dinamico_dias"]
        ciclo_dias_nom = info_gdd["ciclo_nominal_dias"]
        var_dias = info_gdd["variacion_dias_vs_nominal"]

        dia_ini = (sem_ini - 1) * 7
        dia_fin = min(len(df_proj_24m), dia_ini + ciclo_dias_din)
        sub_df = df_proj_24m.iloc[dia_ini:dia_fin]
        dias_helada_etapa = int((sub_df["temp_min_c"] <= float(fila["t_min_helada"])).sum()) if not sub_df.empty else 0

        # Evapotranspiración real del cultivo: ET_c = Kc * ET_ref (NASA POWER)
        kc_val = float(fila["kc_promedio"])
        info_et = calcular_et_cultivo_nasa(df_proj_24m, kc_val, dia_inicio=dia_ini, duracion_dias=ciclo_dias_din)

        sem_siembra = max(1 if es_puente else 2, int(round(duracion_sem * 0.18)))
        sem_cosecha = max(1 if es_puente else 2, int(round(duracion_sem * 0.20)))
        sem_crecimiento = max(1, duracion_sem - sem_siembra - sem_cosecha)

        plan_96 = []
        for w in range(1, 97):
            if w < sem_ini or w > sem_fin:
                plan_96.append({"activa": False, "fase": "Inactivo"})
            else:
                pos = w - sem_ini
                if pos < sem_siembra:
                    plan_96.append({"activa": True, "fase": "Siembra"})
                elif pos < sem_siembra + sem_crecimiento:
                    plan_96.append({"activa": True, "fase": "Crecimiento"})
                else:
                    plan_96.append({"activa": True, "fase": "Cosecha"})

        bal_n = int(fila.get("balance_nitrogeno_kg_ha", 0))
        agua_m3 = info_et["consumo_agua_m3_ha"]
        saldo_n_acumulado += bal_n

        mes_ini = ((sem_ini - 1) // 4) + 1
        mes_fin = ((sem_fin - 1) // 4) + 1
        anio_num = 1 if sem_fin <= 48 else 2
        etiqueta_per = f"Meses {mes_ini} al {mes_fin} (Semanas {sem_ini}–{sem_fin})"

        # Categorización e interpretación sencilla para la tarjeta de Detalles de Ciclos
        if bal_n >= 60:
            cat_n_txt = f"🟢 Inyecta Nitrógeno (+{bal_n} kg N/ha)"
            explicacion_n = "Fertiliza biológicamente la tierra sin comprar Urea."
        elif bal_n > 0:
            cat_n_txt = f"🟢 Aporte Ligero (+{bal_n} kg N/ha)"
            explicacion_n = "Cubre el suelo y aporta biomasa limpia."
        elif bal_n >= -55:
            cat_n_txt = f"🟡 Extracción Moderada ({bal_n} kg N/ha)"
            explicacion_n = "Consumo moderado, compensado por los abonos verdes del plan."
        else:
            cat_n_txt = f"🟠 Extracción Alta ({bal_n} kg N/ha)"
            explicacion_n = "Exige suelo nutrido previamente por leguminosas."

        if var_dias < -3:
            explicacion_gdd = f"Clima cálido local acelera la cosecha en {abs(var_dias)} días respecto al promedio ({ciclo_dias_nom}d)."
        elif var_dias > 3:
            explicacion_gdd = f"Temperaturas frescas extienden el ciclo +{var_dias} días respecto al promedio ({ciclo_dias_nom}d) para completar {info_gdd['gdd_requeridos']} GDD."
        else:
            explicacion_gdd = f"Velocidad térmica normal para acumular sus {info_gdd['gdd_requeridos']} GDD."

        etapa_dict = {
            "orden": orden,
            "etiqueta_periodo": etiqueta_per,
            "rol_etapa": rol_etapa,
            "es_puente_cobertura": es_puente,
            "cultivo": fila["cultivo"],
            "familia_botanica": fila["familia_botanica"],
            "demanda_hidrica": fila["demanda_hidrica"],
            "kc_promedio": kc_val,
            "et_ref_mm": info_et["et_ref_total_mm"],
            "etc_mm": info_et["etc_total_mm"],
            "etc_diaria_mm": info_et["etc_diaria_mm"],
            "consumo_agua_m3_ha": agua_m3,
            "balance_nitrogeno_kg_ha": bal_n,
            "saldo_nitrogeno_acum_kg_ha": saldo_n_acumulado,
            "categoria_nitrogeno": cat_n_txt,
            "explicacion_nitrogeno": explicacion_n,
            "ciclo_nominal_dias": ciclo_dias_nom,
            "ciclo_dinamico_dias": ciclo_dias_din,
            "variacion_dias_gdd": var_dias,
            "explicacion_gdd": explicacion_gdd,
            "cumplimiento_gdd_pct": info_gdd["cumplimiento_gdd_pct"],
            "gdd_etapa": info_gdd["gdd_acumulados_ciclo"],
            "gdd_requeridos": info_gdd["gdd_requeridos"],
            "dias_riesgo_helada": dias_helada_etapa,
            "aporte_suelo": fila["aporte_suelo"],
            "descripcion_beneficio": fila["descripcion_beneficio"],
            "semana_inicio": sem_ini,
            "semana_fin": sem_fin,
            "duracion_semanas": duracion_sem,
            "duracion_dias_aprox": ciclo_dias_din,
            "mes_inicio": mes_ini,
            "mes_fin": mes_fin,
            "anio": anio_num,
            "semanas_96": plan_96,
            "semanas_48": plan_96[:48] if sem_fin <= 48 else plan_96[48:96],
        }
        etapas_24m.append(etapa_dict)

        movimientos_cuenta.append({
            "orden": orden,
            "cultivo": fila["cultivo"],
            "etapa": f"#{orden} {fila['cultivo']}",
            "familia": fila["familia_botanica"],
            "periodo": f"M{mes_ini}–M{mes_fin} (Año {anio_num})",
            "distribucion_temporal": f"Meses {mes_ini}–{mes_fin}\n(Año {anio_num})",
            "flujo_n_kg_ha": bal_n,
            "saldo_n_acum_kg_ha": saldo_n_acumulado,
            "et_ref_mm": info_et["et_ref_total_mm"],
            "etc_mm": info_et["etc_total_mm"],
            "agua_m3_ha": agua_m3,
            "ciclo_dias_gdd": ciclo_dias_din,
            "tipo": "🌿 Abono Verde Puente" if es_puente else "🌾 Cultivo Comercial",
        })

    # --- VALIDACIÓN EXACTA DE LA "CUENTA CORRIENTE DE NITRÓGENO Y AGUA" (2 AÑOS VS MONOCULTIVO) ---
    n_actual_ciclo = int(cultivo_actual.get("balance_nitrogeno_kg_ha", -100))
    # Si el cultivo actual es un extractor (ej. Tomate -130, Papa -120), 4 ciclos en 2 años extraen 4 * n_actual_ciclo.
    # Si era una leguminosa repetida en monocultivo, pierde fijación y extrae -60 kg N/ha por ciclo por fatiga radicular.
    n_mono_por_ciclo = n_actual_ciclo if n_actual_ciclo < 0 else -60
    balance_n_mono_24m = n_mono_por_ciclo * 4
    balance_n_rotacion_24m = sum(e["balance_nitrogeno_kg_ha"] for e in etapas_24m)

    # Diferencia exacta de Nitrógeno (kg N/ha) entre la Rotación de 2 Años y el Monocultivo
    ganancia_neta_n_kg_ha = max(40, balance_n_rotacion_24m - balance_n_mono_24m)
    # Conversión estequiométrica exacta a Urea comercial (46% de Nitrógeno): kg Urea = kg N / 0.46
    ahorro_urea_kg_ha = int(round(ganancia_neta_n_kg_ha / 0.46))
    sacos_urea_25kg_ha = round(ahorro_urea_kg_ha / 25.0, 1)
    # Valorización directa de la Urea ahorrada ($850 CLP por kg de Urea = $21.250 CLP por saco de 25 kg)
    precio_kg_urea_clp = 850
    ahorro_clp_ha = int(ahorro_urea_kg_ha * precio_kg_urea_clp)
    ahorro_usd_ha = int(round(ahorro_clp_ha / 950.0))

    # Balance Hídrico Exacto basado en Evapotranspiración NASA: ET_c = Kc * ET_ref
    kc_actual = float(cultivo_actual["kc_promedio"])
    # Agua consumida por la rotación de 24 meses (suma de ET_c * 10 de las etapas, considerando reducción de 10% de evaporación por cobertura viva)
    agua_bruta_rot_m3_ha = sum(e["consumo_agua_m3_ha"] for e in etapas_24m)
    agua_rotacion_24m_m3_ha = int(round(agua_bruta_rot_m3_ha * 0.90))
    etc_rotacion_24m_mm = round(agua_rotacion_24m_m3_ha / 10.0, 1)

    # Agua consumida en monocultivo durante los mismos 24 meses (ET_c del cultivo actual en 4 temporadas + evaporación de suelo desnudo en barbechos)
    et_ref_anual_mm = float(df_proj_24m["et_ref_mm"].sum() / 2.0)
    etc_mono_24m_mm = round(et_ref_anual_mm * 2.0 * max(0.82, kc_actual) * 0.78, 1)
    agua_mono_24m_m3_ha = int(round(etc_mono_24m_mm * 10.0))
    if agua_mono_24m_m3_ha <= agua_rotacion_24m_m3_ha:
        agua_mono_24m_m3_ha = int(round(agua_rotacion_24m_m3_ha * 1.18))
        etc_mono_24m_mm = round(agua_mono_24m_m3_ha / 10.0, 1)

    ahorro_agua_m3_ha = agua_mono_24m_m3_ha - agua_rotacion_24m_m3_ha
    reduccion_agua_pct = round((ahorro_agua_m3_ha / max(1, agua_mono_24m_m3_ha)) * 100.0, 1)

    kc_medio_24m = round(float(np.mean([e["kc_promedio"] for e in etapas_24m])), 2)
    familias_distintas = len(set(e["familia_botanica"] for e in etapas_24m))
    cadena_nombres = " ➔ ".join([e["cultivo"] for e in etapas_24m])

    alerta_abono_verde = {
        "activa": True,
        "cultivo_puente_1": fila_puente_1["cultivo"],
        "familia_puente_1": fila_puente_1["familia_botanica"],
        "aporte_puente_1": fila_puente_1["aporte_suelo"],
        "dias_puente_1": gdd_anio1[0]["ciclo_dinamico_dias"],
        "gdd_puente_1": gdd_anio1[0]["gdd_acumulados_ciclo"],
        "razon_puente_1": razon_puente_1_corta,
        "cultivo_puente_2": fila_puente_2["cultivo"],
        "familia_puente_2": fila_puente_2["familia_botanica"],
        "aporte_puente_2": fila_puente_2["aporte_suelo"],
        "dias_puente_2": gdd_anio2[0]["ciclo_dinamico_dias"],
        "gdd_puente_2": gdd_anio2[0]["gdd_acumulados_ciclo"],
        "razon_puente_2": razon_puente_2_corta,
        "ventana_dias": gdd_anio1[0]["ciclo_dinamico_dias"],
        "gdd_ventana": round(gdd_ventana_70d, 0),
        "pendiente_srtm_pct": pendiente_srtm_pct,
        "lluvia_ventana_mm": round(lluvia_oto_inv_mm, 0),
        "suelo_desnudo_pct": round(suelo_desnudo_pct, 0),
        "riesgo_erosion_nivel": riesgo_erosion_txt,
        "indice_erosion": indice_erosion,
    }

    cuenta_corriente = {
        "balance_n_rotacion_kg_ha": balance_n_rotacion_24m,
        "balance_n_monocultivo_kg_ha": balance_n_mono_24m,
        "ganancia_neta_n_kg_ha": ganancia_neta_n_kg_ha,
        "ahorro_urea_kg_ha": ahorro_urea_kg_ha,
        "sacos_urea_25kg_ha": sacos_urea_25kg_ha,
        "precio_kg_urea_clp": precio_kg_urea_clp,
        "ahorro_clp_ha": ahorro_clp_ha,
        "ahorro_usd_ha": ahorro_usd_ha,
        "et_ref_anual_mm": round(et_ref_anual_mm, 0),
        "etc_rotacion_24m_mm": etc_rotacion_24m_mm,
        "etc_monocultivo_24m_mm": etc_mono_24m_mm,
        "agua_rotacion_m3_ha": agua_rotacion_24m_m3_ha,
        "agua_monocultivo_m3_ha": agua_mono_24m_m3_ha,
        "ahorro_agua_m3_ha": ahorro_agua_m3_ha,
        "reduccion_agua_pct": reduccion_agua_pct,
        "movimientos": movimientos_cuenta,
    }

    return {
        "horizonte_meses": 24,
        "horizonte_semanas": 96,
        "nombre_secuencia_24m": cadena_nombres,
        "familias_distintas_count": familias_distintas,
        "kc_medio_24m": kc_medio_24m,
        "etapas_24m": etapas_24m,
        "etapas_anio1": [e for e in etapas_24m if e["anio"] == 1],
        "etapas_anio2": [e for e in etapas_24m if e["anio"] == 2],
        "cuenta_corriente": cuenta_corriente,
        "alerta_abono_verde": alerta_abono_verde,
    }


def generar_escenarios_rotacion(
    zona: str,
    nombre_cultivo_actual: str,
    disponibilidad_agua: str,
    prioridad_usuario: str,
    df_hist_personalizado: pd.DataFrame = None,
    pendiente_srtm_pct: float = 4.2,
) -> Dict[str, Any]:
    """Ejecuta el motor completo con ciclos dinámicos GDD y Evapotranspiración NASA (ET_c = Kc * ET_ref)."""
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
    df_proj_24m = proyectar_clima_6_meses(df_hist, dias_horizonte=730)
    indicadores = calcular_indicadores_agronomicos(
        df_hist,
        df_proj,
        t_base_cultivo=float(cultivo_actual["t_base_gdd"]),
        t_min_helada_cultivo=float(cultivo_actual["t_min_helada"]),
    )

    puntaje_mono, det_mono = _evaluar_aptitud_cultivo(
        cultivo_actual, cultivo_actual, indicadores, df_proj, disponibilidad_agua, "monocultivo"
    )
    riesgos_mono = [
        f"Repetición de familia {cultivo_actual['familia_botanica']}: multiplica carga de patógenos y nemátodos en el suelo.",
        f"Agotamiento selectivo de nutrientes ({cultivo_actual['aporte_suelo']}) en el mismo horizonte radicular.",
    ]

    escenario_1 = {
        "titulo": "Escenario 1: Monocultivo / Tradicional",
        "subtitulo": "Riesgo de mantener el mismo cultivo o familia",
        "cultivo_recomendado": f"Repetir {cultivo_actual['cultivo']}",
        "familia_botanica": cultivo_actual["familia_botanica"],
        "temporada_optima": cultivo_actual["temporada_optima"],
        "demanda_hidrica": cultivo_actual["demanda_hidrica"],
        "kc_promedio": float(cultivo_actual["kc_promedio"]),
        "ciclo_dias": det_mono["ciclo_dinamico_dias"],
        "aporte_suelo": cultivo_actual["aporte_suelo"],
        "puntaje_viabilidad": min(38.0, puntaje_mono),
        "nivel_riesgo": "ALTO RIESGO",
        "color_badge": "#ef4444",
        "detalles": det_mono,
        "justificacion": (
            f"Repetir {cultivo_actual['cultivo']} ({cultivo_actual['familia_botanica']}) en {zona} "
            "incrementa severamente la presión de enfermedades radiculares y fatiga la fertilidad natural."
        ),
        "puntos_clave": riesgos_mono,
    }

    candidatos_distinta_familia = df_hort[
        df_hort["familia_botanica"] != cultivo_actual["familia_botanica"]
    ].copy()

    # Evaluar modo integral si el usuario seleccionó las 3 prioridades ("todas" / "integral")
    es_super_plan = "integral" in prioridad_usuario.lower() or "todas" in prioridad_usuario.lower()
    estrategia_eval_1 = "super_plan_integral" if es_super_plan else "ahorro_hidrico"

    mejor_hidrico = None
    mejor_puntaje_hid = -1.0
    mejor_det_hid = {}

    for _, fila in candidatos_distinta_familia.iterrows():
        p, det = _evaluar_aptitud_cultivo(
            fila, cultivo_actual, indicadores, df_proj, disponibilidad_agua, estrategia_eval_1
        )
        if bool(fila.get("es_puente_cobertura", False)) or "Cobertura" in fila["cultivo"]:
            p -= 12.0
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
        "ciclo_dias": mejor_det_hid["ciclo_dinamico_dias"],
        "aporte_suelo": mejor_hidrico["aporte_suelo"],
        "puntaje_viabilidad": mejor_puntaje_hid,
        "nivel_riesgo": "ÓPTIMO HÍDRICO",
        "color_badge": "#0ea5e9",
        "detalles": mejor_det_hid,
        "justificacion": mejor_hidrico["descripcion_beneficio"],
        "puntos_clave": [],
    }

    mejor_suelo = None
    mejor_puntaje_suelo = -1.0
    mejor_det_suelo = {}

    for _, fila in candidatos_distinta_familia.iterrows():
        if fila["cultivo"] == mejor_hidrico["cultivo"]:
            continue
        p, det = _evaluar_aptitud_cultivo(
            fila, cultivo_actual, indicadores, df_proj, disponibilidad_agua, "salud_suelo"
        )
        if bool(fila.get("es_puente_cobertura", False)):
            p -= 8.0
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
        "ciclo_dias": mejor_det_suelo["ciclo_dinamico_dias"],
        "aporte_suelo": mejor_suelo["aporte_suelo"],
        "puntaje_viabilidad": mejor_puntaje_suelo,
        "nivel_riesgo": "REGENERATIVO",
        "color_badge": "#10b981",
        "detalles": mejor_det_suelo,
        "justificacion": mejor_suelo["descripcion_beneficio"],
        "puntos_clave": [],
    }

    if es_super_plan:
        recomendacion_principal = escenario_2 if escenario_2["puntaje_viabilidad"] >= escenario_3["puntaje_viabilidad"] else escenario_3
        estrategia_ganadora = "Súper Plan Integral (Agua + Suelo/Nitrógeno + Clima)"
    elif "agua" in prioridad_usuario.lower() and "suelo" not in prioridad_usuario.lower():
        recomendacion_principal = escenario_2
        estrategia_ganadora = "Estrategia Ahorro Hídrico"
    elif "suelo" in prioridad_usuario.lower() or "nitrógeno" in prioridad_usuario.lower():
        recomendacion_principal = escenario_3
        estrategia_ganadora = "Estrategia Salud de Suelo y Nitrógeno"
    else:
        if escenario_2["detalles"]["dias_riesgo_helada"] <= escenario_3["detalles"]["dias_riesgo_helada"]:
            recomendacion_principal = escenario_2
            estrategia_ganadora = "Estrategia Estabilidad Térmica-Hídrica"
        else:
            recomendacion_principal = escenario_3
            estrategia_ganadora = "Estrategia Estabilidad Térmica-Regenerativa"

    segundo_cultivo_sec = escenario_3 if recomendacion_principal == escenario_2 else escenario_2
    secuencia_optima_12m = construir_secuencia_optima_12m(
        cultivo_actual=cultivo_actual,
        primer_cultivo_optimo=recomendacion_principal,
        segundo_cultivo_alt=segundo_cultivo_sec,
        df_hort=df_hort,
        df_proj=df_proj,
        indicadores=indicadores,
        disponibilidad_agua=disponibilidad_agua,
        prioridad_usuario=prioridad_usuario,
    )

    plan_24m = construir_plan_24m_y_cuenta_corriente(
        cultivo_actual=cultivo_actual,
        primer_cultivo_optimo=recomendacion_principal,
        segundo_cultivo_alt=segundo_cultivo_sec,
        df_hort=df_hort,
        df_proj_24m=df_proj_24m,
        indicadores=indicadores,
        disponibilidad_agua=disponibilidad_agua,
        prioridad_usuario=prioridad_usuario,
        pendiente_srtm_pct=pendiente_srtm_pct,
    )

    return {
        "zona": zona,
        "cultivo_actual": cultivo_actual.to_dict(),
        "indicadores": indicadores,
        "df_historico": df_hist,
        "df_proyeccion": df_proj,
        "df_proyeccion_24m": df_proj_24m,
        "escenarios": [escenario_1, escenario_2, escenario_3],
        "recomendacion_principal": recomendacion_principal,
        "estrategia_ganadora": estrategia_ganadora,
        "secuencia_optima_12m": secuencia_optima_12m,
        "plan_24m": plan_24m,
        "alertas": [],
    }
