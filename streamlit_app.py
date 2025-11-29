import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ================================================================
# CONFIGURACIÓN INICIAL
# ================================================================

st.set_page_config(
    page_title="Predicción de Demanda Educativa con Gemini",
    page_icon="",
    layout="centered"
)

st.title(" Predicción de Demanda Educativa con Gemini 2.5")
st.write("Sube tu dataset y genera proyecciones inteligentes.")


# ============================
# 1. CONFIGURAR GEMINI API KEY
# ============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# ================================================================
# 2. SUBIR ARCHIVO CSV
# ================================================================

st.subheader("📁 Cargar datos históricos")

archivo = st.file_uploader("Sube el archivo CSV con los datos históricos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.write("### Vista previa de los datos")
    st.dataframe(df)

    # Lista de programas detectados
    nombre_col_programa = "programa"
    nombre_col_anio = "anio"
    nombre_col_demanda = "demanda"

    if nombre_col_programa in df.columns:

        programas = df[nombre_col_programa].unique().tolist()

        st.success("Datos cargados correctamente.")
    else:
        st.error("Tu CSV debe tener la columna 'programa'.")
else:
    st.stop()


# ================================================================
# 3. SELECCIÓN DEL USUARIO
# ================================================================

st.subheader("Configurar Predicción")

programa_usuario = st.selectbox("Selecciona el programa a proyectar:", programas)
años = st.slider("¿Cuántos años deseas proyectar?", 1, 20, 5)
tendencias_usuario = st.text_area(
    "Describe tendencias globales, sociales, tecnológicas o locales que puedan impactar la demanda educativa:",
    placeholder="Ej. demanda de IA, crecimiento en TI, necesidad de competencias digitales…"
)


# ================================================================
# 4. GESTIÓN DEL PROMPT EXPERTO
# ================================================================

def generar_prediccion(programa, años, tendencias, datos_resumen):
    """
    Llama a Gemini usando un prompt experto en predicción de demanda educativa.
    """

    prompt = f"""
Eres un **experto senior en estadística educativa, modelado de series de tiempo, 
análisis laboral y planeación universitaria**, con 20 años de experiencia.

Tu misión: generar una **predicción cuantitativa y estratégica** combinando:
1) Los datos históricos reales proporcionados  
2) Conocimiento general externo que tú sabes sobre tendencias educativas  
Sin inventar cifras específicas no sustentadas.

----------------------------------------------------

----------------------------------------------------

---

## PROGRAMA A ANALIZAR
- Programa académico: **{programa}**
- Años a proyectar: **{años}**

---

## DATOS HISTÓRICOS DISPONIBLES
{datos_resumen}

---

🌍 **TENDENCIAS EXTERNAS A CONSIDERAR**
El usuario indicó:

{tendencias if tendencias.strip() != "" else "No se proporcionaron tendencias adicionales."}

Úsalas solo como ajustes cualitativos, nunca como sustituto de los datos reales.

----------------------------------------------------
**INSTRUCCIONES DEL ANÁLISIS**

### 1. Analizar los datos históricos
- Tendencia general  
- Crecimiento promedio anual (CAGR)  
- Estacionalidad o patrones  
- Ruido o variabilidad  
- Outliers o puntos anómalos  

### 2. Integrar tendencias externas
Puedes usar conocimiento general sobre:
- Tendencias globales de educación superior  
- Comportamiento de matrícula en programas similares  
- Cambios demográficos y tecnológicos  
- Demanda laboral del área del programa  
SIN inventar números externos exactos.

Explica cómo afectan la proyección.

### 3. Producir la proyección numérica
- Proyecta año por año  
- Números coherentes basados en el historial  
- Ajustes suaves inspirados en tendencias globales  

### 4. Formato obligatorio de la respuesta

#### 🔹 **1. Análisis estadístico del historial**
Texto claro y técnico.

#### 🔹 **2. Factores externos relevantes**
Tendencias generales, sin cifras inventadas.

#### 🔹 **3. Tabla de proyección (año → demanda esperada)**

Ejemplo:
Año | Demanda estimada  
----|------------------  
2025 | X  
2026 | X  

#### 🔹 **4. Supuestos del modelo**
Justificación técnica del método usado.

#### 🔹 **5. Conclusión ejecutiva**
Clara, objetiva y accionable.

----------------------------------------------------
## FORMATO FINAL
Responde de manera ordenada, con buena estructura profesional.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")

    respuesta = model.generate_content(prompt)

    return respuesta.text


# ================================================================
# 5. BOTÓN PARA GENERAR PREDICCIÓN
# ================================================================

if st.button("🚀 Generar Predicción"):
    # Filtrar datos para ese programa
    df_filtrado = df[df[nombre_col_programa] == programa_usuario]

    # Crear mini resumen del CSV para enviarlo a Gemini
    resumen = df_filtrado.head(20).to_string(index=False)

    with st.spinner("Generando análisis con Gemini..."):
        resultado = generar_prediccion(programa_usuario, años, tendencias_usuario, resumen)

    st.subheader("📈 Resultado de la Predicción")
    st.write(resultado)

    st.success("Predicción generada correctamente.")
