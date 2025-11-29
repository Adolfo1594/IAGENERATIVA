import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ================================================================
# INICIALIZACIÓN DE VARIABLES DE SESIÓN
# ================================================================
# Guarda el análisis generado
if "resultado_prediccion" not in st.session_state:
    st.session_state.resultado_prediccion = None

# Guarda historial de preguntas
if "historial_preguntas" not in st.session_state:
    st.session_state.historial_preguntas = []

# ================================================================
# CONFIGURACIÓN INICIAL DE LA APP
# ================================================================
st.set_page_config(
    page_title="Predicción de Demanda Educativa con Gemini",
    page_icon="",
    layout="centered"
)

st.title("Predicción de Demanda Educativa con Gemini ")
st.write("Sube tu dataset y genera proyecciones inteligentes basadas en datos reales y tendencias educativas.")

# ================================================================
# CONFIGURACIÓN DE GEMINI
# ================================================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("No se encontró la API Key en st.secrets. Asegúrate de configurarla en Streamlit Cloud.")
    st.stop()

# ================================================================
# 1. SUBIR ARCHIVO CSV
# ================================================================
st.subheader("Cargar datos históricos")

archivo = st.file_uploader("Sube el archivo CSV con los datos históricos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.write("### Vista previa de los datos cargados:")
    st.dataframe(df)

    # Columnas obligatorias
    nombre_col_programa = "programa"
    nombre_col_anio = "anio"
    nombre_col_demanda = "demanda"

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
st.subheader("⚙️ Configurar Predicción")

programa_usuario = st.selectbox("Selecciona el programa a proyectar:", programas)
años = st.slider("¿Cuántos años deseas proyectar?", 1, 20, 5)

tendencias_usuario = st.text_area(
    "Describe tendencias que puedan impactar la demanda educativa:",
    placeholder="Ej. crecimiento de IA, digitalización, nuevas regulaciones, cambios demográficos…"
)

# ================================================================
# 3. FUNCIÓN PRINCIPAL PARA GENERAR LA PREDICCIÓN
# ================================================================
def generar_prediccion(programa, años, tendencias, datos_resumen):
    prompt = f"""
Eres un **experto senior en estadística educativa, modelado de series de tiempo, 
análisis laboral y planeación universitaria**, con 20 años de experiencia.

Tu misión: generar una **predicción cuantitativa y estratégica** combinando:
1) Los datos históricos reales proporcionados  
2) Conocimiento general externo que tú sabes sobre tendencias educativas  
Sin inventar cifras específicas no sustentadas.

----------------------------------------------------
----------------------------------------------------
PROGRAMA
- {programa}
AÑOS A PROYECTAR
- {años}

----------------------------------------------------
DATOS HISTÓRICOS (RESUMEN)
{datos_resumen}

----------------------------------------------------
TENDENCIAS EXTERNAS
{tendencias if tendencias.strip() else "No se indicaron tendencias adicionales"}

----------------------------------------------------
INSTRUCCIONES

1. Analiza los datos:
- tendencia general
- crecimiento promedio anual
- cambios bruscos
- anomalías

2. Integra conocimiento general del sector educativo.

3. Genera una proyección por año:
Año | Demanda Estimada  
----|------------------

4. Incluye:
- Análisis estadístico
- Factores externos
- Tabla de proyección
- Supuestos del modelo
- Conclusión ejecutiva

Responde de manera estructurada, profesional y clara.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    respuesta = model.generate_content(prompt)
    return respuesta.text

# ================================================================
# 4. BOTÓN PARA GENERAR PREDICCIÓN
# ================================================================
if st.button("Generar Predicción"):

    df_filtrado = df[df[nombre_col_programa] == programa_usuario]
    resumen = df_filtrado.head(20).to_string(index=False)

    with st.spinner("Generando análisis con Gemini..."):
        resultado = generar_prediccion(programa_usuario, años, tendencias_usuario, resumen)

    # Guardar análisis en sesión
    st.session_state.resultado_prediccion = resultado
    st.session_state.historial_preguntas = []  # Resetear historial si se genera un nuevo análisis

    st.success("Predicción generada correctamente.")

# ================================================================
# 5. MOSTRAR ANÁLISIS SI YA EXISTE
# ================================================================
if st.session_state.resultado_prediccion:
    st.subheader("Resultado de la Predicción")
    st.write(st.session_state.resultado_prediccion)

# ================================================================
# 6. SECCIÓN DE PREGUNTAS ADICIONALES
# ================================================================
st.subheader("Haz preguntas sobre el análisis generado")

if st.session_state.resultado_prediccion:
    pregunta = st.text_input("Escribe tu pregunta:")

    if st.button("Responder pregunta"):
        if pregunta.strip() == "":
            st.warning("Escribe una pregunta antes de continuar.")
        else:
            prompt_pregunta = f"""
Aquí tienes el análisis generado previamente:

------------------------------------------------
{st.session_state.resultado_prediccion}
------------------------------------------------

PREGUNTA DEL USUARIO:
{pregunta}

Responde de forma clara, útil y consistente con el análisis previo.
"""

            model = genai.GenerativeModel("gemini-2.0-flash")
            respuesta = model.generate_content(prompt_pregunta).text

            # Guardar en historial
            st.session_state.historial_preguntas.append((pregunta, respuesta))

# ================================================================
# 7. MOSTRAR HISTORIAL DE PREGUNTAS
# ================================================================
if st.session_state.historial_preguntas:
    st.subheader("Preguntas y respuestas adicionales")

    for i, (p, r) in enumerate(st.session_state.historial_preguntas):
        st.write(f"### ❓ Pregunta {i+1}")
        st.write(p)
        st.write("**Respuesta:**")
        st.write(r)
        st.write("---")
