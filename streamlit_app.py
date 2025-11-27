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
    page_icon="📊",
    layout="centered"
)

st.title("📊 Predicción de Demanda Educativa con Gemini 2.5")
st.write("Sube tu dataset y genera proyecciones inteligentes.")


# ================================================================
# 1. CONFIGURAR API KEY
# ================================================================

st.subheader("🔐 Configurar API Key")

api_key = st.text_input("Ingresa tu API Key de Gemini:", type="password")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Por favor ingresa tu API Key antes de continuar.")


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

st.subheader("🎯 Configurar Predicción")

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
Eres un **experto senior en analítica educativa, modelado estadístico, 
predicción de demanda académica y planeación estratégica universitaria**, 
con 20 años de experiencia asesorando instituciones de educación superior.

Tu objetivo: elaborar una **proyección de demanda estudiantil realista y con base analítica**.

---

## 📘 PROGRAMA A ANALIZAR
- Programa académico: **{programa}**
- Años a proyectar: **{años}**

---

## 📊 DATOS HISTÓRICOS DISPONIBLES
{datos_resumen}

---

## 🌍 TENDENCIAS A CONSIDERAR
El usuario indicó estas tendencias externas que pueden impactar la demanda:
➡️ {tendencias}

---

## 🧠 INSTRUCCIONES DE ANÁLISIS
Debes:

### 1. Analizar los datos históricos
- Identificar patrones, estacionalidades o quiebres.
- Calcular crecimiento promedio.
- Detectar anomalías significativas.

### 2. Integrar el contexto externo
- Relaciona las tendencias con el comportamiento del programa.
- Explica su impacto en la demanda.

### 3. Producir predicción cuantitativa
- Proyecta demanda año por año.
- Utiliza lógica coherente, estadística cualitativa y análisis contextual.
- **Evita inventar números aleatorios.**

### 4. Entregar una respuesta clara con estos bloques:
1. **Tabla de proyección año → demanda estimada**  
2. **Análisis detallado de la proyección**  
3. **Conclusión ejecutiva**  
4. **Recomendaciones estratégicas para la institución**  

---

## 📤 FORMATO FINAL
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
