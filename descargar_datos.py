import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# Configuración de directorios
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Coordenadas representativas del Valle Central de Chile
ZONAS_CHILE = {
    "Quillota": {
        "lat": -32.8833,
        "lon": -71.2500,
        "region": "Valparaíso",
        "temp_base": 16.5,
        "lluvia_anual": 350,
    },
    "Rancagua": {
        "lat": -34.1708,
        "lon": -70.7444,
        "region": "O'Higgins",
        "temp_base": 15.0,
        "lluvia_anual": 500,
    },
    "Talca": {
        "lat": -35.4264,
        "lon": -71.6554,
        "region": "Maule",
        "temp_base": 14.5,
        "lluvia_anual": 700,
    },
    "Chillan": {
        "lat": -36.6066,
        "lon": -72.1034,
        "region": "Ñuble",
        "temp_base": 13.5,
        "lluvia_anual": 950,
    },
}

VARIABLES_NASA = [
    "T2M",      # Temperatura media a 2m (°C)
    "T2M_MAX",  # Temperatura máxima a 2m (°C)
    "T2M_MIN",  # Temperatura mínima a 2m (°C)
    "TS",       # Temperatura del suelo (°C)
    "PRECTOTCORR",  # Precipitación corregida (mm/día)
    "RH2M",  # Humedad relativa a 2m (%)
    "EVLAND", # evapotranspiración
    "GWETTOP",  # Humedad suelo superficial (fracción 0-1)
    "GWETROOT",  # Humedad suelo zona radicular (fracción 0-1)
    "GWETPROF", # humedad perfil completo
    "ALLSKY_SFC_SW_DWN",  # Radiación solar incidente (MJ/m^2/día o kWh/m^2/día)
]

MAPEO_COLUMNAS = {
    "T2M": "temp_media_c",
    "T2M_MAX": "temp_max_c",
    "T2M_MIN": "temp_min_c",
    "TS": "temp_suelo_c",
    "PRECTOTCORR": "precipitacion_mm",
    "RH2M": "humedad_relativa_pct",
    "EVLAND": "evapotranspiracion_mm",
    "GWETTOP": "humedad_suelo_sup",
    "GWETROOT": "humedad_suelo_raiz",
    "GWETPROF": "humedad_suelo_prof",
    "ALLSKY_SFC_SW_DWN": "radiacion_solar",
}


def limpiar_csv_nasa(texto_csv: str, nombre_zona: str) -> pd.DataFrame:
    """Detecta dinámicamente el fin del encabezado (-END HEADER-) de NASA POWER,

    construye la columna de fecha, reemplaza valores nulos (-999) e interpola.
    """
    lineas = texto_csv.splitlines()
    indice_inicio = 0
    for idx, linea in enumerate(lineas):
        if "-END HEADER-" in linea.upper() or "END HEADER" in linea.upper():
            indice_inicio = idx + 1
            break

    contenido_datos = "\n".join(lineas[indice_inicio:])
    df = pd.read_csv(io.StringIO(contenido_datos))

    # Normalizar nombres de columnas por seguridad
    df.columns = [col.strip().upper() for col in df.columns]

    # Construir columna 'fecha' según el formato devuelto por NASA POWER (YEAR+DOY o YEAR+MO+DY)
    if {"YEAR", "MO", "DY"}.issubset(df.columns):
        df["fecha"] = pd.to_datetime(
            df[["YEAR", "MO", "DY"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day"}
            )
        )
    elif {"YEAR", "DOY"}.issubset(df.columns):
        df["fecha"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["DOY"].astype(str),
            format="%Y-%j",
        )
    else:
        raise ValueError(
            f"Columnas de fecha no reconocidas en el CSV de NASA POWER: {df.columns.tolist()}"
        )

    # Reemplazar valores centinela de error (-999, -999.0, -99) por NaN
    df = df.replace([-999, -999.0, -99, -99.0], np.nan)

    # Renombrar columnas al español
    df = df.rename(columns=MAPEO_COLUMNAS)
    columnas_finales = ["fecha"] + list(MAPEO_COLUMNAS.values())
    df = df[[c for c in columnas_finales if c in df.columns]].copy()

    # Ordenar por fecha e interpolar valores faltantes (común en los últimos días del año en curso)
    df = df.sort_values("fecha").reset_index(drop=True)
    columnas_numericas = [c for c in df.columns if c != "fecha"]
    df[columnas_numericas] = (
        df[columnas_numericas]
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
    )

    # Respaldo en caso de que T2M no venga o tenga vacíos
    if "temp_media_c" not in df.columns:
        df["temp_media_c"] = (
            (df["temp_max_c"] + df["temp_min_c"]) / 2.0
        ).round(2)
    df["zona"] = nombre_zona

    return df


def generar_datos_sinteticos(
    nombre_zona: str,
    info_zona: dict,
    fecha_inicio: str = "2021-01-01",
    fecha_fin: str = "2026-10-31",
) -> pd.DataFrame:
    """Generador de respaldo (Offline Fallback) que simula el clima mediterráneo

    del Valle Central de Chile (veranos secos/cálidos, inviernos húmedos/fríos).
    """
    semilla = abs(hash(nombre_zona)) % (2**32)
    rng = np.random.default_rng(semilla)

    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
    n = len(fechas)
    dia_anio = fechas.dayofyear.to_numpy()

    # Estacionalidad hemisferio sur: máximo térmico en enero (día ~15), mínimo en julio (día ~196)
    onda_estacional = np.cos(2 * np.pi * (dia_anio - 15) / 365.25)

    t_base = info_zona["temp_base"]
    temp_media = (
        t_base + 6.5 * onda_estacional + rng.normal(0, 1.8, n)
    )
    amplitud_termica = (
        13.0 + 3.5 * onda_estacional + rng.normal(0, 1.5, n)
    )
    amplitud_termica = np.clip(amplitud_termica, 5.0, 22.0)

    temp_max = np.round(temp_media + amplitud_termica / 2.0, 2)
    temp_min = np.round(temp_media - amplitud_termica / 2.0, 2)
    temp_suelo = np.round(temp_media + 1.2 * onda_estacional + rng.normal(0, 1.0, n), 2)

    # Precipitación concentrada en otoño-invierno (mayo a agosto)
    prob_lluvia = np.clip(0.18 - 0.16 * onda_estacional, 0.01, 0.45)
    hay_lluvia = rng.random(n) < prob_lluvia
    escala_lluvia = (info_zona["lluvia_anual"] / 500.0) * 12.0
    precipitacion = np.where(
        hay_lluvia, rng.exponential(scale=escala_lluvia, size=n), 0.0
    )
    precipitacion = np.round(precipitacion, 2)

    # Humedad relativa, radiación solar y evapotranspiración
    humedad_relativa = np.clip(
        68.0 - 18.0 * onda_estacional + rng.normal(0, 6.0, n),
        25.0,
        98.0,
    )
    radiacion_solar = np.clip(
        18.0 + 11.0 * onda_estacional + rng.normal(0, 2.0, n),
        3.0,
        32.0,
    )
    evapotranspiracion = np.clip(
        2.2 + 1.8 * onda_estacional + rng.normal(0, 0.4, n),
        0.1,
        8.5,
    )

    # Humedad de suelo (GWETTOP, GWETROOT y GWETPROF) con memoria hídrica y ligera tendencia de sequía
    tendencia_sequia = np.linspace(0.0, -0.05, n)
    base_humedad_raiz = (
        0.52
        - 0.20 * onda_estacional
        + (info_zona["lluvia_anual"] - 500) / 3000.0
        + tendencia_sequia
    )
    aporte_lluvia = (
        pd.Series(precipitacion).rolling(14, min_periods=1).sum().to_numpy()
        * 0.0025
    )

    humedad_raiz = np.clip(
        base_humedad_raiz + aporte_lluvia + rng.normal(0, 0.02, n),
        0.15,
        0.92,
    )
    humedad_sup = np.clip(
        humedad_raiz - 0.05 * onda_estacional + rng.normal(0, 0.03, n),
        0.10,
        0.95,
    )
    humedad_prof = np.clip(
        humedad_raiz + 0.03 + rng.normal(0, 0.01, n),
        0.20,
        0.95,
    )

    df = pd.DataFrame(
        {
            "fecha": fechas,
            "temp_media_c": np.round(temp_media, 2),
            "temp_max_c": temp_max,
            "temp_min_c": temp_min,
            "temp_suelo_c": temp_suelo,
            "precipitacion_mm": precipitacion,
            "humedad_relativa_pct": np.round(humedad_relativa, 1),
            "evapotranspiracion_mm": np.round(evapotranspiracion, 2),
            "humedad_suelo_sup": np.round(humedad_sup, 3),
            "humedad_suelo_raiz": np.round(humedad_raiz, 3),
            "humedad_suelo_prof": np.round(humedad_prof, 3),
            "radiacion_solar": np.round(radiacion_solar, 2),
            "zona": nombre_zona,
        }
    )
    return df


def descargar_o_generar_zona(
    nombre_zona: str,
    info_zona: dict,
    inicio: str = "20210101",
    fin: str = "20261031",
) -> pd.DataFrame:
    """Consulta la API de NASA POWER en formato CSV; si falla o no hay red,

    activa automáticamente el generador sintético offline.
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(VARIABLES_NASA),
        "community": "AG",
        "longitude": info_zona["lon"],
        "latitude": info_zona["lat"],
        "start": inicio,
        "end": fin,
        "format": "CSV",
    }

    ruta_salida = DATA_DIR / f"nasa_{nombre_zona.lower()}.csv"

    try:
        print(
            f"[NASA POWER] Descargando datos para {nombre_zona} ({info_zona['lat']}, {info_zona['lon']})..."
        )
        respuesta = requests.get(url, params=params, timeout=15)
        respuesta.raise_for_status()
        df = limpiar_csv_nasa(respuesta.text, nombre_zona)
        origen = "API NASA POWER"
    except Exception as exc:
        print(
            f"  [!] No se pudo conectar con NASA POWER ({exc}). Generando respaldo sintético realista para {nombre_zona}..."
        )
        df = generar_datos_sinteticos(nombre_zona, info_zona)
        origen = "Respaldo Sintético Offline"

    df.to_csv(ruta_salida, index=False)
    print(
        f"  [OK] Guardado en '{ruta_salida}' ({len(df)} registros diarios | Origen: {origen})"
    )
    return df


def main():
    print("=" * 65)
    print(
        " INICIANDO DESCARGA Y LIMPIEZA DE DATOS AGROCLIMÁTICOS NASA POWER"
    )
    print("=" * 65)
    for nombre_zona, info_zona in ZONAS_CHILE.items():
        descargar_o_generar_zona(nombre_zona, info_zona)
    print("\nProceso finalizado. Archivos CSV listos para trabajar 100% OFFLINE en ./data/")


if __name__ == "__main__":
    main()
