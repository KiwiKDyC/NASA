from typing import Dict, Any, Tuple, Optional, List
import math
import numpy as np
import pandas as pd
import requests

from descargar_datos import ZONAS_CHILE
from modelo_rotacion import cargar_datos_zona


def calcular_distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fórmula de Haversine para calcular la distancia en kilómetros entre dos coordenadas."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def extraer_coordenadas_mapa(salida_mapa: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, str, Optional[float]]]:
    """Extrae (lat, lon, tipo_seleccion, area_aprox_ha) desde la respuesta de st_folium.
    Soporta tanto el dibujo de un polígono/rectángulo/marcador (last_active_drawing)
    como un clic directo sobre el mapa (last_clicked).
    """
    if not salida_mapa or not isinstance(salida_mapa, dict):
        return None

    # 1. Priorizar polígono, rectángulo o marcador recién dibujado por el agricultor
    dibujo = salida_mapa.get("last_active_drawing")
    if dibujo and isinstance(dibujo, dict) and "geometry" in dibujo:
        geom = dibujo["geometry"]
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Point" and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            return (round(lat, 5), round(lon, 5), "Punto marcado en el predio", None)

        elif gtype == "Polygon" and coords and len(coords[0]) >= 3:
            anillo = coords[0]
            lons = [ float(p[0]) for p in anillo ]
            lats = [ float(p[1]) for p in anillo ]
            # Centroide simple del polígono
            lat_c = float(np.mean(lats[:-1])) if len(lats) > 3 else float(np.mean(lats))
            lon_c = float(np.mean(lons[:-1])) if len(lons) > 3 else float(np.mean(lons))

            # Estimar superficie aproximada en hectáreas (proyección plana local)
            lat_rad = math.radians(lat_c)
            x_m = [ (lon - lon_c) * 111320.0 * math.cos(lat_rad) for lon in lons ]
            y_m = [ (lat - lat_c) * 110540.0 for lat in lats ]
            area_m2 = 0.5 * abs(
                sum(x_m[i] * y_m[i + 1] - x_m[i + 1] * y_m[i] for i in range(len(x_m) - 1))
            )
            area_ha = round(max(0.1, area_m2 / 10000.0), 2)
            return (round(lat_c, 5), round(lon_c, 5), f"Parcela dibujada (~{area_ha} ha)", area_ha)

    # 2. Si no hay polígono activo, usar el último clic directo sobre el mapa
    clic = salida_mapa.get("last_clicked")
    if clic and isinstance(clic, dict) and "lat" in clic and "lng" in clic:
        return (
            round(float(clic["lat"]), 5),
            round(float(clic["lng"]), 5),
            "Clic directo en el potrero",
            None,
        )

    return None


def geocodificar_inverso_ligero(lat: float, lon: float, zona_cercana: str, dist_km: float) -> str:
    """Obtiene el nombre de la localidad/comuna mediante Nominatim OpenStreetMap (timeout corto);
    si estamos 100% offline, construye una referencia geográfica clara usando la estación más cercana.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": 12,
            "accept-language": "es",
        }
        headers = {"User-Agent": "AgroSateliteNASA-SpaceAppsChile/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            addr = data.get("address", {})
            localidad = (
                addr.get("municipality")
                or addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
            )
            region = addr.get("state", ZONAS_CHILE[zona_cercana]["region"])
            if localidad:
                return f"{localidad}, {region} (Nodo satelital {zona_cercana} a {dist_km:.1f} km)"
    except Exception:
        pass

    if dist_km <= 8.0:
        return f"Sector {zona_cercana} (Región de {ZONAS_CHILE[zona_cercana]['region']})"
    return f"Predio rural a {dist_km:.1f} km de {zona_cercana} (Región de {ZONAS_CHILE[zona_cercana]['region']})"


def interpolar_dataframe_espacial(lat: float, lon: float) -> Tuple[pd.DataFrame, str, float, Dict[str, float]]:
    """Interpola ponderadamente por inverso de la distancia (IDW) las series diarias
    de nuestras 4 zonas calibradas (Quillota, Rancagua, Talca, Chillán) para cualquier
    coordenada (lat, lon) del Valle Central, aplicando además un ajuste microclimático
    por latitud y continentalidad/costa.
    """
    distancias = {}
    for nombre_z, info_z in ZONAS_CHILE.items():
        d = calcular_distancia_km(lat, lon, info_z["lat"], info_z["lon"])
        distancias[nombre_z] = max(0.5, d)

    zona_mas_cercana = min(distancias, key=distancias.get)
    dist_min_km = distancias[zona_mas_cercana]

    # Si está muy cerca de una estación base (< 5 km), usar directamente esa estación
    if dist_min_km <= 5.0:
        df_base = cargar_datos_zona(zona_mas_cercana).copy()
        pesos = {z: (1.0 if z == zona_mas_cercana else 0.0) for z in ZONAS_CHILE}
        return df_base, zona_mas_cercana, round(dist_min_km, 1), pesos

    # Pesos IDW (Inverse Distance Weighting con potencia p=2)
    inv_d2 = {z: 1.0 / (d ** 2) for z, d in distancias.items()}
    suma_inv = sum(inv_d2.values())
    pesos = {z: val / suma_inv for z, val in inv_d2.items()}

    dfs = {z: cargar_datos_zona(z) for z in ZONAS_CHILE}
    df_ref = dfs[zona_mas_cercana].copy()

    cols_numericas = [
        c
        for c in df_ref.columns
        if c not in ["fecha", "zona", "SUELO_DESNUDO"] and pd.api.types.is_numeric_dtype(df_ref[c])
    ]

    for col in cols_numericas:
        serie_interp = np.zeros(len(df_ref), dtype=float)
        for z, p in pesos.items():
            if col in dfs[z].columns:
                serie_interp += dfs[z][col].to_numpy(dtype=float) * p
        df_ref[col] = np.round(serie_interp, 3 if ("humedad_suelo" in col or col in ["NDVI", "EVI", "NDMI"]) else 2)

    # Ajuste microclimático suave según longitud (costa vs precordillera) respecto a la estación más cercana
    lon_ref = ZONAS_CHILE[zona_mas_cercana]["lon"]
    delta_lon = lon - lon_ref  # positivo hacia la cordillera, negativo hacia la costa
    ajuste_termico = np.clip(-delta_lon * 1.1, -1.8, 1.8)
    df_ref["temp_media_c"] = np.round(df_ref["temp_media_c"] + ajuste_termico, 2)
    df_ref["temp_min_c"] = np.round(df_ref["temp_min_c"] + ajuste_termico * 1.15, 2)
    df_ref["temp_max_c"] = np.round(df_ref["temp_max_c"] + ajuste_termico * 0.85, 2)

    if "NDVI" in df_ref.columns and "humedad_suelo_sup" in df_ref.columns:
        df_ref["SUELO_DESNUDO"] = np.where(
            (df_ref["NDVI"] < 0.32) | ((df_ref["NDVI"] < 0.36) & (df_ref["humedad_suelo_sup"] < 0.35)),
            1,
            0,
        )

    df_ref["zona"] = zona_mas_cercana
    return df_ref, zona_mas_cercana, round(dist_min_km, 1), pesos


def caracterizar_parcela_automatica(
    lat: float,
    lon: float,
    tipo_seleccion: str = "Ubicación en mapa",
    area_ha: Optional[float] = None,
) -> Dict[str, Any]:
    """Ejecuta la caracterización Zero-Input de una coordenada o polígono en el mapa:
    1. Identifica localidad/comuna e interpola las variables NASA POWER + Sentinel-2.
    2. Auto-infiere Disponibilidad de Agua ('Baja', 'Media', 'Alta') y Prioridad Agronómica
       sin que el agricultor tenga que responder preguntas técnicas.
    3. Genera la 'Ficha de Diagnóstico Automático del Predio'.
    """
    df_parcela, zona_cercana, dist_km, pesos_idw = interpolar_dataframe_espacial(lat, lon)
    nombre_ubicacion = geocodificar_inverso_ligero(lat, lon, zona_cercana, dist_km)

    ultimos_30 = df_parcela.tail(30)
    ultimos_365 = df_parcela.tail(365)

    hum_raiz_pct = float(ultimos_30["humedad_suelo_raiz"].mean() * 100.0)
    hum_sup_pct = float(ultimos_30["humedad_suelo_sup"].mean() * 100.0)
    temp_suelo_c = float(ultimos_30["temp_suelo_c"].mean()) if "temp_suelo_c" in ultimos_30.columns else float(ultimos_30["temp_media_c"].mean())
    precip_anual_mm = float(ultimos_365["precipitacion_mm"].sum())
    ndvi_actual = float(ultimos_30["NDVI"].mean()) if "NDVI" in ultimos_30.columns else 0.52
    ndmi_actual = float(ultimos_30["NDMI"].mean()) if "NDMI" in ultimos_30.columns else 0.15
    suelo_desnudo_pct = float(ultimos_365["SUELO_DESNUDO"].mean() * 100.0) if "SUELO_DESNUDO" in ultimos_365.columns else 25.0
    dias_helada_anual = int((ultimos_365["temp_min_c"] <= 0.5).sum())

    # --- REGLA 1: AUTO-INFERENCIA DE DISPONIBILIDAD DE AGUA ---
    indice_hidrico_compuesto = (
        0.55 * hum_raiz_pct
        + 0.25 * min(100.0, (precip_anual_mm / 700.0) * 100.0)
        + 0.20 * ((ndmi_actual + 0.35) * 100.0)
    )
    if hum_raiz_pct < 54.0 or precip_anual_mm < 380.0 or indice_hidrico_compuesto < 48.0:
        agua_inferida = "Baja"
        etiqueta_humedad = f"Media-Baja ({hum_raiz_pct:.0f}%)"
    elif hum_raiz_pct >= 64.5 and precip_anual_mm >= 580.0:
        agua_inferida = "Alta"
        etiqueta_humedad = f"Alta ({hum_raiz_pct:.0f}%)"
    else:
        agua_inferida = "Media"
        etiqueta_humedad = f"Media ({hum_raiz_pct:.0f}%)"

    # --- REGLA 2: AUTO-INFERENCIA DE COBERTURA Y RIESGO DE HELADAS ---
    if suelo_desnudo_pct >= 28.0 or ndvi_actual < 0.38:
        estado_cobertura = f"Suelo parcialmente desnudo ({suelo_desnudo_pct:.0f}% del año)"
    elif ndvi_actual >= 0.62:
        estado_cobertura = f"Cobertura vegetal alta (NDVI {ndvi_actual:.2f})"
    else:
        estado_cobertura = f"Cobertura vegetal moderada (NDVI {ndvi_actual:.2f})"

    if dias_helada_anual >= 12:
        riesgo_heladas_txt = f"Alto ({dias_helada_anual} días fríos/año)"
    elif dias_helada_anual >= 4:
        riesgo_heladas_txt = f"Moderado ({dias_helada_anual} días fríos/año)"
    else:
        riesgo_heladas_txt = "Bajo (Clima templado estable)"

    # --- REGLA 3: AUTO-INFERENCIA DE PRIORIDAD AGRONÓMICA ---
    if agua_inferida == "Baja" or ndmi_actual < 0.12:
        prioridad_inferida = "💧 Ahorrar agua"
        motivo_prioridad = "Se detectó limitación hídrica o baja lluvia acumulada en el sector."
    elif dias_helada_anual >= 10 or temp_suelo_c < 13.5:
        prioridad_inferida = "☀️ Estabilidad térmica (evitar heladas)"
        motivo_prioridad = "Se detectaron noches frías frecuentes en este sector geográfico."
    elif suelo_desnudo_pct >= 22.0 or ndvi_actual < 0.52:
        prioridad_inferida = "🤎 Recuperar salud del suelo / nitrógeno"
        motivo_prioridad = "Se detectó desgaste de cobertura y ventanas de suelo descubierto."
    else:
        prioridad_inferida = "🤎 Recuperar salud del suelo / nitrógeno"
        motivo_prioridad = "Buena disponibilidad hídrica; óptimo para regenerar nitrógeno y cortar plagas."

    resumen_ficha = (
        f"📍 Ubicación: {nombre_ubicacion} | "
        f"💧 Humedad de suelo detectada: {etiqueta_humedad} | "
        f"🌿 Estado cobertura: {estado_cobertura} | "
        f"❄️ Riesgo de heladas: {riesgo_heladas_txt}"
    )

    return {
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "zona_cercana": zona_cercana,
        "distancia_estacion_km": dist_km,
        "nombre_ubicacion": nombre_ubicacion,
        "tipo_seleccion": tipo_seleccion,
        "area_ha": area_ha,
        "agua_inferida": agua_inferida,
        "prioridad_inferida": prioridad_inferida,
        "motivo_prioridad": motivo_prioridad,
        "etiqueta_humedad": etiqueta_humedad,
        "humedad_raiz_pct": round(hum_raiz_pct, 1),
        "humedad_sup_pct": round(hum_sup_pct, 1),
        "temp_suelo_c": round(temp_suelo_c, 1),
        "precip_anual_mm": round(precip_anual_mm, 0),
        "ndvi_actual": round(ndvi_actual, 2),
        "ndmi_actual": round(ndmi_actual, 2),
        "suelo_desnudo_pct": round(suelo_desnudo_pct, 0),
        "riesgo_heladas_txt": riesgo_heladas_txt,
        "estado_cobertura": estado_cobertura,
        "resumen_ficha": resumen_ficha,
        "df_interpolado": df_parcela,
        "pesos_idw": pesos_idw,
    }
