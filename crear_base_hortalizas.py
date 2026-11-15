from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def generar_catalogo_hortalizas() -> pd.DataFrame:
    """Construye la base de conocimiento agronómico para hortalizas comerciales y
    cultivos de cobertura (abonos verdes de ventana corta) representativos del Valle
    Central de Chile (Valparaíso a Biobío), incluyendo balance de Nitrógeno (kg N/ha),
    consumo hídrico estacional (m3/ha) y diversidad de familias botánicas:
    Solanaceae, Fabaceae, Brassicaceae, Amaryllidaceae, Asteraceae, Poaceae,
    Cucurbitaceae y Apiaceae.
    """
    datos_hortalizas = [
        {
            "cultivo": "Papa",
            "familia_botanica": "Solanaceae (Solanáceas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 7.0,
            "gdd_requeridos": 1350,
            "t_min_helada": -1.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.85,
            "consumo_agua_m3_ha": 5200,
            "balance_nitrogeno_kg_ha": -120,
            "ciclo_dias": 120,
            "es_puente_cobertura": False,
            "aporte_suelo": "Extractor alto (-120 kg N/ha)",
            "descripcion_beneficio": (
                "Alta rentabilidad y buena estructura de suelo por aporque, pero extrae alto "
                "potasio/nitrógeno (-120 kg N/ha) y comparte nemátodos y tizón con otras Solanaceae."
            ),
        },
        {
            "cultivo": "Tomate",
            "familia_botanica": "Solanaceae (Solanáceas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1500,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Alta",
            "kc_promedio": 0.95,
            "consumo_agua_m3_ha": 6400,
            "balance_nitrogeno_kg_ha": -130,
            "ciclo_dias": 125,
            "es_puente_cobertura": False,
            "aporte_suelo": "Extractor alto (-130 kg N/ha)",
            "descripcion_beneficio": (
                "Cultivo estrella de verano en el Valle Central; altamente sensible a heladas "
                "y agotador de nutrientes (-130 kg N/ha). Requiere biofumigación y rotación con Fabaceae."
            ),
        },
        {
            "cultivo": "Pimentón",
            "familia_botanica": "Solanaceae (Solanáceas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1450,
            "t_min_helada": 1.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.88,
            "consumo_agua_m3_ha": 5400,
            "balance_nitrogeno_kg_ha": -100,
            "ciclo_dias": 130,
            "es_puente_cobertura": False,
            "aporte_suelo": "Extractor moderado (-100 kg N/ha)",
            "descripcion_beneficio": (
                "Alto valor comercial con demanda hídrica moderada bajo riego por goteo, "
                "pero muy susceptible a Phytophthora y nemátodos si se repite tras Solanaceae."
            ),
        },
        {
            "cultivo": "Poroto verde",
            "familia_botanica": "Fabaceae (Leguminosas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 9.0,
            "gdd_requeridos": 950,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.78,
            "consumo_agua_m3_ha": 3900,
            "balance_nitrogeno_kg_ha": 65,
            "ciclo_dias": 85,
            "es_puente_cobertura": False,
            "aporte_suelo": "Fijador de Nitrógeno (+65 kg N/ha)",
            "descripcion_beneficio": (
                "Fija nitrógeno atmosférico mediante simbiosis con Rhizobium (+65 kg N/ha), "
                "recuperando suelos agotados por tomate, papa o maíz en un ciclo corto."
            ),
        },
        {
            "cultivo": "Haba / Arveja",
            "familia_botanica": "Fabaceae (Leguminosas)",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 900,
            "t_min_helada": -4.5,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.70,
            "consumo_agua_m3_ha": 3100,
            "balance_nitrogeno_kg_ha": 80,
            "ciclo_dias": 110,
            "es_puente_cobertura": False,
            "aporte_suelo": "Fijador de Nitrógeno (+80 kg N/ha)",
            "descripcion_beneficio": (
                "Aprovecha lluvias invernales reduciendo riego artificial, tolera heladas "
                "y aporta +80 kg N/ha biológicos antes de los cultivos de primavera."
            ),
        },
        {
            "cultivo": "Brócoli / Coliflor",
            "familia_botanica": "Brassicaceae (Brasicáceas)",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 5.0,
            "gdd_requeridos": 1000,
            "t_min_helada": -3.5,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.80,
            "consumo_agua_m3_ha": 4100,
            "balance_nitrogeno_kg_ha": -65,
            "ciclo_dias": 95,
            "es_puente_cobertura": False,
            "aporte_suelo": "Biofumigante glucosinolatos (-65 kg N/ha)",
            "descripcion_beneficio": (
                "Sus residuos ricos en glucosinolatos actúan como biofumigante natural contra "
                "hongos de suelo (Verticillium y Rhizoctonia), excelente tras Fabaceae."
            ),
        },
        {
            "cultivo": "Repollo",
            "familia_botanica": "Brassicaceae (Brasicáceas)",
            "temporada_optima": "Otoño-Primavera",
            "t_base_gdd": 4.5,
            "gdd_requeridos": 980,
            "t_min_helada": -4.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.76,
            "consumo_agua_m3_ha": 3700,
            "balance_nitrogeno_kg_ha": -55,
            "ciclo_dias": 100,
            "es_puente_cobertura": False,
            "aporte_suelo": "Supresor de malezas y hongos (-55 kg N/ha)",
            "descripcion_beneficio": (
                "Alta rusticidad frente a heladas invernales, cobertura densa que frena la "
                "erosión hídrica y corta el ciclo de nemátodos de las Solanaceae."
            ),
        },
        {
            "cultivo": "Cebolla",
            "familia_botanica": "Amaryllidaceae (Aliáceas)",
            "temporada_optima": "Otoño-Primavera",
            "t_base_gdd": 5.0,
            "gdd_requeridos": 1200,
            "t_min_helada": -4.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.80,
            "consumo_agua_m3_ha": 4300,
            "balance_nitrogeno_kg_ha": -50,
            "ciclo_dias": 135,
            "es_puente_cobertura": False,
            "aporte_suelo": "Sanitizante radicular (-50 kg N/ha)",
            "descripcion_beneficio": (
                "Sus exudados azufrados reducen hongos de suelo (Fusarium/Verticillium). "
                "Excelente tolerancia al frío y raíz superficial que deja buena estructura."
            ),
        },
        {
            "cultivo": "Ajo",
            "familia_botanica": "Amaryllidaceae (Aliáceas)",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 1100,
            "t_min_helada": -5.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.72,
            "consumo_agua_m3_ha": 3400,
            "balance_nitrogeno_kg_ha": -40,
            "ciclo_dias": 130,
            "es_puente_cobertura": False,
            "aporte_suelo": "Fungicida natural alicina (-40 kg N/ha)",
            "descripcion_beneficio": (
                "Baja huella hídrica estacional y altísima resistencia a heladas; libera "
                "compuestos sulfurados que desinfectan biológicamente la rizósfera."
            ),
        },
        {
            "cultivo": "Lechuga",
            "familia_botanica": "Asteraceae (Asteráceas)",
            "temporada_optima": "Todo el año (Óptimo Otoño-Primavera)",
            "t_base_gdd": 4.5,
            "gdd_requeridos": 750,
            "t_min_helada": -2.5,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.72,
            "consumo_agua_m3_ha": 2900,
            "balance_nitrogeno_kg_ha": -35,
            "ciclo_dias": 75,
            "es_puente_cobertura": False,
            "aporte_suelo": "Extractor ligero (-35 kg N/ha)",
            "descripcion_beneficio": (
                "Ciclo corto y menor huella hídrica total por cosecha; ideal para rotación "
                "rápida que corta el ciclo de plagas de Solanaceae y Brassicaceae."
            ),
        },
        {
            "cultivo": "Maíz dulce",
            "familia_botanica": "Poaceae (Gramíneas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 8.0,
            "gdd_requeridos": 1250,
            "t_min_helada": 0.0,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.84,
            "consumo_agua_m3_ha": 4800,
            "balance_nitrogeno_kg_ha": -75,
            "ciclo_dias": 105,
            "es_puente_cobertura": False,
            "aporte_suelo": "Aporte de carbono y rastrojo (-75 kg N/ha)",
            "descripcion_beneficio": (
                "Sistema radicular fasciculado denso que mejora la porosidad y aporta abundante "
                "rastrojo lignificado para elevar la materia orgánica tras leguminosas."
            ),
        },
        {
            "cultivo": "Zapallo",
            "familia_botanica": "Cucurbitaceae (Cucurbitáceas)",
            "temporada_optima": "Primavera-Verano",
            "t_base_gdd": 10.0,
            "gdd_requeridos": 1400,
            "t_min_helada": 0.5,
            "demanda_hidrica": "Media",
            "kc_promedio": 0.82,
            "consumo_agua_m3_ha": 4600,
            "balance_nitrogeno_kg_ha": -65,
            "ciclo_dias": 115,
            "es_puente_cobertura": False,
            "aporte_suelo": "Cobertura viva anti-evaporación (-65 kg N/ha)",
            "descripcion_beneficio": (
                "Su follaje rastrero sombrea el suelo, disminuye la evaporación directa "
                "de humedad superficial (GWETTOP) y suprime malezas de forma natural."
            ),
        },
        {
            "cultivo": "Zanahoria",
            "familia_botanica": "Apiaceae (Apiáceas)",
            "temporada_optima": "Otoño-Primavera",
            "t_base_gdd": 5.0,
            "gdd_requeridos": 1050,
            "t_min_helada": -3.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.75,
            "consumo_agua_m3_ha": 3500,
            "balance_nitrogeno_kg_ha": -45,
            "ciclo_dias": 105,
            "es_puente_cobertura": False,
            "aporte_suelo": "Descompactador biológico (-45 kg N/ha)",
            "descripcion_beneficio": (
                "Raíz pivotante profunda que mejora la infiltración de agua (GWETROOT), "
                "baja demanda hídrica estacional y excelente tolerancia a bajas temperaturas."
            ),
        },
        {
            "cultivo": "Vicia (Abono Verde)",
            "familia_botanica": "Fabaceae (Leguminosas)",
            "temporada_optima": "Puente Corto Otoño-Invierno (60-80 días)",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 480,
            "t_min_helada": -6.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.58,
            "consumo_agua_m3_ha": 1800,
            "balance_nitrogeno_kg_ha": 90,
            "ciclo_dias": 70,
            "es_puente_cobertura": True,
            "aporte_suelo": "Inyección biológica de Nitrógeno (+90 kg N/ha)",
            "descripcion_beneficio": (
                "Cultivo puente de ventana corta (60-80 días, Vicia atropurpurea/sativa): cubre el "
                "suelo contra la erosión de lluvias e inyecta +90 kg N/ha equivalentes a ~195 kg de Urea."
            ),
        },
        {
            "cultivo": "Mostaza Blanca (Biofumigante)",
            "familia_botanica": "Brassicaceae (Brasicáceas)",
            "temporada_optima": "Puente Corto Post-Verano (60-75 días)",
            "t_base_gdd": 4.5,
            "gdd_requeridos": 450,
            "t_min_helada": -4.5,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.60,
            "consumo_agua_m3_ha": 1750,
            "balance_nitrogeno_kg_ha": 30,
            "ciclo_dias": 65,
            "es_puente_cobertura": True,
            "aporte_suelo": "Biofumigación contra nemátodos y hongos (+30 kg N/ha)",
            "descripcion_beneficio": (
                "Abono verde biofumigante rápido (Sinapis alba, 60-75 días): libera isotiocianatos al "
                "incorporarse, controlando nemátodos y hongos dejados por Solanaceae (Tomate/Papa)."
            ),
        },
        {
            "cultivo": "Avena + Vicia (Cobertura)",
            "familia_botanica": "Poaceae + Fabaceae",
            "temporada_optima": "Otoño-Invierno",
            "t_base_gdd": 4.0,
            "gdd_requeridos": 720,
            "t_min_helada": -6.0,
            "demanda_hidrica": "Baja",
            "kc_promedio": 0.62,
            "consumo_agua_m3_ha": 2300,
            "balance_nitrogeno_kg_ha": 75,
            "ciclo_dias": 85,
            "es_puente_cobertura": True,
            "aporte_suelo": "Fijador de Nitrógeno (+75 kg N/ha) y Cobertura anti-erosión",
            "descripcion_beneficio": (
                "Mezcla forrajera/cobertura: aporta estructura radicular (Poaceae) + fijación de "
                "nitrógeno (+75 kg N/ha, Fabaceae), maximizando infiltración y eliminando suelo desnudo."
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
                "balance_nitrogeno_kg_ha",
                "consumo_agua_m3_ha",
                "ciclo_dias",
                "es_puente_cobertura",
            ]
        ].to_string(index=False)
    )
