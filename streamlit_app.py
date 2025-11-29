import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ================================================================
# INICIALIZACIÓN DE VARIABLES DE SESIÓN
# ================================================================
# Se usa para guardar la predicción después de generarla
if "resultado_prediccion" not in st.session_state:
    st.session_state.resultado_prediccion = None

# ================================================================
# CONFIGURACIÓN INICIAL DE LA APP
# ================================================================
st.set_page_config(
    page_title="Predicción de Demanda Educativa con Gemini",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Predicción de Demanda Educativa con Gemini 2.5")
st.write("Sube tu dataset y genera proyecciones inteligentes basadas en datos reales y tendencias educativas.")

# ================================================================
# CONFIGURACIÓN DE GEMINI (USANDO SECRETS)
# ================================================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("No se encontró la API Key en st.secrets. Asegúrate de configurarla en Streamlit Cloud.")
    st.stop()

# ================================================================
# 1. SUBIR ARCHIVO CSV
# ================================================================
st.subheader("📁 Cargar datos históricos")

archivo = st.file_uploader("Sube el archivo CSV con los datos históricos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.write("### Vista previa de los datos cargados:")
    st.dataframe(df)

    # Columnas obligatorias
    nombre_col_programa = "programa"
    nombre_col_anio = "anio"
    nombre_col_demanda = "demanda"

    # Validación
    if nombre_col_programa not in df.columns:
        st.error("El CSV debe contener la columna 'programa'.")
        st.stop()

    programas = df[nombre_col_programa].unique().tolist()
else:
    st.info("Sube un archivo CSV para continuar.")
    st.stop()

# ================================================================
# 2. SELECCIÓN DEL USUARIO
# ================================================================
st.subheader("🎯 Configurar Predicción")

programa_usuario = st.selectbox("Selecciona el programa a proyectar:", programas)
años = st.slider("¿Cuántos años deseas proyectar?", 1, 20, 5)

tendencias_usuario = st.text_area(
    "Describe tendencias globales, sociales, tecnológicas o locales que puedan impactar la demanda educativa:",
    placeholder="Ej. crecimiento de IA, digitalización, nuevas regulaciones, cambios demográficos…"
)

# ================================================================
# 3. FUNCIÓN PRINCIPAL PARA GENERAR LA PREDICCIÓN
# ================================================================
def generar_prediccion(programa, años, tendencias, datos_resumen):
    prompt = f"""
Eres un **especialista senior en estadística educativa, proyecciones de matrícula,
planeación universitaria y análisis laboral**, con más de 20 años de experiencia 
asesorando instituciones de educación superior.

Tu tarea es generar una **proyección de demanda educativa precisa, objetiva y basada en datos**, 
combinando:

1) Datos históricos proporcionados  
2) Conocimiento general de tendencias globales del sector educativo  
3) Patrones de comportamiento típicos en programas académicos similares  

NO inventes datos externos exactos; usa lógica estadística, inferencia y análisis experto.

----------------------------------------------------
📘 **PROGRAMA A ANALIZAR**
- Programa: {programa}
- Años a proyectar: {años}

----------------------------------------------------
📊 **DATOS HISTÓRICOS (RESUMEN)**
{datos_resumen}

----------------------------------------------------
🌍 **TENDENCIAS EXTERNAS INDICADAS POR EL USUARIO**
{tendencias if tendencias.strip() else "No se proporcionaron tendencias adicionales."}

Úsalas como moduladores cualitativos, no como cifras exactas.

----------------------------------------------------
🧠 **INSTRUCCIONES DEL ANÁLISIS**

### 1. Analiza los datos históricos:
- Identifica tendencia general
- Calcula crecimiento promedio anual
- Revisa variaciones o quiebres
- Reconoce estacionalidad o patrones
- Detecta outliers o anomalías

### 2. Integra conocimiento experto externo:
- Tendencias globales de educación superior
- Demanda laboral del área del programa
- Comportamientos típicos de matrícula
- Cambios demográficos o tecnológicos

### 3. Genera la proyección:
- Año por año
- Basada en crecimiento histórico + ajuste cualitativo por tendencias
- Evita saltos bruscos o incoherentes

### 4. El formato de respuesta DEBE incluir:

#### 🔹 1. Análisis estadístico del historial
Explicación clara basada en datos reales.

#### 🔹 2. Factores externos relevantes (sin cifras inventadas)

#### 🔹 3. **Tabla de proyección**
Año | Demanda Estimada  
----|------------------

#### 🔹 4. Supuestos del modelo
Qué se asumió y por qué.

#### 🔹 5. Conclusión ejecutiva
Recomendaciones concretas para la institución.

----------------------------------------------------
Responde de manera profesional, estructurada y clara.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    respuesta = model.generate_content(prompt)
    return respuesta.text

# ================================================================
# 4. BOTÓN PARA GENERAR PREDICCIÓN
# ================================================================
if st.button("🚀 Generar Predicción"):

    df_filtrado = df[df[nombre_col_programa] == programa_usuario]
    resumen = df_filtrado.head(20).to_string(index=False)

    with st.spinner("Generando análisis con Gemini..."):
        resultado = generar_prediccion(programa_usuario, años, tendencias_usuario, resumen)

    # Guardar en session_state
    st.session_state.resultado_prediccion = resultado

    st.subheader("📈 Resultado de la Predicción")
    st.write(st.session_state.resultado_prediccion)
    st.success("Predicción generada correctamente.")

# ================================================================
# 5. SECCIÓN DE PREGUNTAS ADICIONALES
# ================================================================
st.subheader("🧠 Haz preguntas sobre el análisis generado")

if st.session_state.resultado_prediccion:
    pregunta = st.text_input("Escribe tu pregunta:")
    
    if st.button("Responder pregunta"):
        prompt_pregunta = f"""
Aquí está el análisis previo que generaste:

------------------------------------------------
{st.session_state.resultado_prediccion}
------------------------------------------------

El usuario pregunta ahora:

❓ {pregunta}

Por favor responde de forma clara, útil y consistente con el análisis original.
Evita contradecir los datos previos.
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        respuesta = model.generate_content(prompt_pregunta)

        st.write("### Respuesta del sistema:")
        st.write(respuesta.text)
else:
    st.info("Genera primero la predicción para activar esta sección.")
