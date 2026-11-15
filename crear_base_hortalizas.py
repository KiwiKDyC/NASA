from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def generar_catalogo_hortalizas() -> pd.DataFrame:
    """Construye la base de conocimiento agronómico para hortalizas y cultivos
    de rotación representativos del Valle Central de Chile (Valparaíso a Biobío).
    """
    datos_hortalizas = [
        {
            "cultivo": "Papa",
            "familia_botanica": "Solanáceas",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 7.0,
            "gdd_requeridos": 1350,
            "t_min_helada": -1.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.85,
            "ciclo_dias": 120,
            "aporte_suelo": "Extractor alto",
            "descripcion_beneficio": (
                "Alta rentabilidad y buena estructura de suelo por aporque, pero extrae alto "
                "potasio/nitrógeno y comparte nemátodos y tizón con otras Solanáceas."
            ),
        },
        {
            "cultivo": "Tomate",
            "familia_botanica": "Solanáceas",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1500,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Alta",
            "kc_promedio": 0.95,
            "ciclo_dias": 125,
            "aporte_suelo": "Extractor alto",
            "descripcion_beneficio": (
                "Cultivo estrella de verano en el Valle Central; altamente sensible a heladas "
                "y agotador de nutrientes. Requiere rotación estricta de al menos 2-3 ciclos."
            ),
        },
        {
            "cultivo": "Pimentón",
            "familia_botanica": "Solanáceas",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1450,
            "t_min_helada": 1.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.88,
            "ciclo_dias": 130,
            "aporte_suelo": "Extractor moderado",
            "descripcion_beneficio": (
                "Alto valor comercial con demanda hídrica moderada bajo riego por goteo, "
                "pero muy susceptible a Phytophthora si se repite tras Solanáceas."
            ),
        },
        {
            "cultivo": "Cebolla",
            "familia_botanica": "Aliáceas",
            "temporada_optima": "Otoño-Primavera",
            "t_base_gdd": 5.0,
            "gdd_requeridos": 1200,
            "t_min_helada": -4.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.80,
            "ciclo_dias": 140,
            "aporte_suelo": "Sanitizante radicular",
            "descripcion_beneficio": (
                "Sus exudados azufrados reducen hongos de suelo (Fusarium/Verticillium). "
                "Excelente tolerancia al frío y raíz superficial que deja buena estructura."
            ),
        },
        {
            "cultivo": "Lechuga",
            "familia_botanica": "Asteráceas",
            "temporada_optima": "Todo el año (Óptimo Otoño-Primavera)",
            "t_base_gdd": 4.5,
            "gdd_requeridos": 750,
            "t_min_helada": -2.5,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.72,
            "ciclo_dias": 75,
            "aporte_suelo": "Extractor ligero",
            "descripcion_beneficio": (
                "Ciclo corto y menor huella hídrica total por cosecha; ideal para rotación "
                "rápida que corta el ciclo de plagas de Solanáceas y Cucurbitáceas."
            ),
        },
        {
            "cultivo": "Poroto verde",
            "familia_botanica": "Fabáceas (Leguminosas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 9.0,
            "gdd_requeridos": 950,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.78,
            "ciclo_dias": 85,
            "aporte_suelo": "Fijador de Nitrógeno",
            "descripcion_beneficio": (
                "Fija nitrógeno atmosférico mediante simbiosis con Rhizobium (40-70 kg N/ha), "
                "recuperando suelos agotados por tomate, papa o maíz en un ciclo corto."
            ),
        },
        {
            "cultivo": "Haba / Arveja",
            "familia_botanica": "Fabáceas (Leguminosas)",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 900,
            "t_min_helada": -4.5,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.70,
            "ciclo_dias": 110,
            "aporte_suelo": "Fijador de Nitrógeno",
            "descripcion_beneficio": (
                "Aprovecha lluvias invernales reduciendo riego artificial, tolera heladas "
                "y aporta abundante nitrógeno y biomasa antes de los cultivos de primavera."
            ),
        },
        {
            "cultivo": "Zapallo",
            "familia_botanica": "Cucurbitáceas",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1400,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.82,
            "ciclo_dias": 120,
            "aporte_suelo": "Cobertura anti-erosión",
            "descripcion_beneficio": (
                "Su follaje rastrero sombrea el suelo, disminuye la evaporación directa "
                "de humedad superficial (GWETTOP) y suprime malezas de forma natural."
            ),
        },
        {
            "cultivo": "Zanahoria",
            "familia_botanica": "Apiáceas",
            "temporada_optima": "Otoño-Primavera",
            "t_base_gdd": 5.0,
            "gdd_requeridos": 1050,
            "t_min_helada": -3.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.75,
            "ciclo_dias": 105,
            "aporte_suelo": "Descompactador biológico",
            "descripcion_beneficio": (
                "Raíz pivotante profunda que mejora la infiltración de agua (GWETROOT), "
                "baja demanda hídrica estacional y excelente tolerancia a bajas temperaturas."
            ),
        },
        {
            "cultivo": "Avena + Vicia (Cobertura)",
            "familia_botanica": "Poáceas + Fabáceas",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 850,
            "t_min_helada": -6.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.62,
            "ciclo_dias": 100,
            "aporte_suelo": "Fijador de Nitrógeno y Cobertura anti-erosión",
            "descripcion_beneficio": (
                "Abono verde biofumigante: aporta materia orgánica, fija nitrógeno, "
                "maximiza retención de humedad radicular y corta al 100% plagas hortícolas."
            ),
        },
    ]

    df = pd.DataFrame(datos_hortalizas)
    ruta_csv = DATA_DIR / "base_hortalizas.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"[OK] Base agronómica generada en '{ruta_csv}' ({len(df)} cultivos).")
    return df


if __name__ == "__main__":
    df_hort = generar_catalogo_hortalizas()
    print(
        df_hort[
            [
                "cultivo",
                "familia_botanica",
                "demanda_hidrica",
                "t_min_helada",
                "aporte_suelo",
            ]
        ].to_string(index=False)
    )
