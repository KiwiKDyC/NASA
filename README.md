# 🛰️ AgroRotación Chile — NASA Space Apps Challenge

Sistema de apoyo a la decisión (Web & Mobile) para recomendar **rotación inteligente de cultivos de hortalizas** en el Valle Central de Chile (**Valparaíso a Biobío**), cruzando datos satelitales agroclimáticos y de suelo de **NASA POWER (2021–2026)** con requerimientos fisiológicos y fitosanitarios de hortalizas.

---

## 🚀 Inicio Rápido para el Equipo (Modo 100% Offline)

Los datos satelitales de las 4 zonas (**Quillota, Rancagua, Talca y Chillán**) y el catálogo agronómico ya están incluidos y limpios en la carpeta [`data/`](./data/), por lo que puedes levantar el proyecto inmediatamente después de clonar:

### 1. Clonar el repositorio
```bash
git clone https://github.com/KiwiKDyC/NASA.git
cd NASA
```

### 2. Crear entorno virtual e instalar dependencias
**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**En Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación en Streamlit
```bash
streamlit run app.py
```

---

## 🗂️ Arquitectura del Proyecto

```text
├── data/                        # Datos listos para ejecución 100% OFFLINE
│   ├── nasa_quillota.csv        # Serie diaria NASA POWER (Valparaíso)
│   ├── nasa_rancagua.csv        # Serie diaria NASA POWER (O'Higgins)
│   ├── nasa_talca.csv           # Serie diaria NASA POWER (Maule)
│   ├── nasa_chillan.csv         # Serie diaria NASA POWER (Ñuble)
│   └── base_hortalizas.csv      # Catálogo agronómico de 10 cultivos/coberturas
├── descargar_datos.py           # ETL API NASA POWER + Limpieza + Respaldo Sintético Offline
├── crear_base_hortalizas.py     # Generador de la matriz agronómica de hortalizas
├── modelo_rotacion.py           # Motor de cálculo (GDD, heladas, proyección 6m y 3 escenarios)
├── app.py                       # Interfaz interactiva en Streamlit + Plotly
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Exclusión de archivos basura/entornos virtuales
└── .gitattributes               # Estandarización de saltos de línea (Windows/Mac/Linux)
```

---

## 🛰️ Variables de NASA POWER Utilizadas (`community=AG`)

| Código NASA | Columna en CSV | Descripción |
| :--- | :--- | :--- |
| `T2M` | `temp_media_c` | Temperatura media del aire a 2 metros (°C) |
| `T2M_MAX` | `temp_max_c` | Temperatura máxima diaria a 2 metros (°C) |
| `T2M_MIN` | `temp_min_c` | Temperatura mínima diaria a 2 metros (°C) |
| `TS` | `temp_suelo_c` | Temperatura superficial del suelo (°C) |
| `PRECTOTCORR` | `precipitacion_mm` | Precipitación diaria corregida MERRA-2 (mm/día) |
| `RH2M` | `humedad_relativa_pct` | Humedad relativa a 2 metros (%) |
| `EVLAND` | `evapotranspiracion_mm` | Evapotranspiración sobre superficie terrestre (mm/día) |
| `GWETTOP` | `humedad_suelo_sup` | Humedad de suelo superficial 0–5 cm (fracción 0–1) |
| `GWETROOT` | `humedad_suelo_raiz` | Humedad de suelo en zona radicular 0–100 cm (fracción 0–1) |
| `GWETPROF` | `humedad_suelo_prof` | Humedad en perfil completo de suelo (fracción 0–1) |
| `ALLSKY_SFC_SW_DWN` | `radiacion_solar` | Radiación solar de onda corta incidente |

---

## 🤝 Flujo de Trabajo Colaborativo en Git (Recomendado para la Hackathon)

Para evitar conflictos de código (`merge conflicts`) durante la hackathon:

1. **Actualizar siempre antes de empezar un cambio:**
   ```bash
   git pull origin main
   ```
2. **Trabajar en ramas por funcionalidad (opcional pero seguro):**
   ```bash
   git checkout -b feat/mejora-interfaz
   ```
3. **Guardar y subir cambios:**
   ```bash
   git add .
   git commit -m "feat: descripción clara del cambio"
   git push -u origin feat/mejora-interfaz
   ```
4. **División de responsabilidades por archivo:**
   - **Frontend / UI / UX:** editar `app.py`
   - **Ciencia de Datos / Reglas de Negocio:** editar `modelo_rotacion.py`
   - **Datos Agronómicos / Nuevos Cultivos:** editar `crear_base_hortalizas.py` y ejecutar `python crear_base_hortalizas.py`
   - **Ingesta NASA POWER / Nuevas Comunas:** editar `descargar_datos.py` y ejecutar `python descargar_datos.py`
