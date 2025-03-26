import streamlit as st
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import re

# Configurar la API de OpenAI
openai.api_key = "KEY"

# Cargar datos con barra de progreso
datos = pd.read_csv("transacciones_sinteticas.csv")

# Obtener ubicaciones únicas antes de codificar
ubicaciones = datos["ubicacion"].unique().tolist()

datos = pd.get_dummies(datos, columns=["metodo_pago", "dispositivo", "ubicacion", "usuario", "direccion_ip"])

progress_bar_carga = st.progress(0, "Cargando datos...")
for i in range(100):
    time.sleep(0.01)
    progress_bar_carga.progress(i + 1)
st.write("Datos cargados")

def entrenar_modelo_ml(datos):
    X = datos.drop("fraude", axis=1)
    y = datos["fraude"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    return modelo

# Entrenar modelo con barra de progreso
progress_bar_entrenamiento = st.progress(0, "Entrenando modelo...")
modelo_ml = entrenar_modelo_ml(datos)
for i in range(100):
    time.sleep(0.02)
    progress_bar_entrenamiento.progress(i + 1)
st.write("Modelo entrenado")

def predecir_fraude_ml(modelo, datos_transaccion):
    datos_df = pd.DataFrame([datos_transaccion], index=[0])
    datos_df = pd.get_dummies(datos_df, columns=["metodo_pago", "dispositivo", "ubicacion", "usuario", "direccion_ip"])

    columnas_modelo = modelo.feature_names_in_
    for col in columnas_modelo:
        if col not in datos_df.columns:
            datos_df[col] = 0
    datos_df = datos_df[columnas_modelo]

    prediccion = modelo.predict(datos_df)
    return prediccion[0]

def analizar_transaccion_openai(datos_transaccion):
    prompt = f"Analiza la siguiente transacción e indica su riesgo de fraude: {datos_transaccion}"
    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Reemplaza con el modelo que prefieras
        messages=[
            {"role": "system", "content": "Eres un experto en detección de fraudes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return respuesta.choices[0].message.content.strip()

def validar_direccion_ip(direccion_ip):
    patron = r"^(\d{1,3}\.){3}\d{1,3}$"
    if re.match(patron, direccion_ip):
        return True
    else:
        return False

# Interfaz en Streamlit
st.title("FraudGuard AI - Detección de Fraudes")
st.write("Ingrese los datos de la transacción para evaluar su nivel de riesgo.")

# Formulario de entrada de datos
monto = st.number_input("Monto de la transacción")
ubicacion = st.selectbox("Ubicación de la transacción", ubicaciones)
usuario = st.text_input("Nombre de usuario")
metodo_pago = st.selectbox("Método de pago", ["tarjeta_credito", "tarjeta_debito", "transferencia", "paypal"])
dispositivo = st.selectbox("Dispositivo", ["PC", "móvil", "tableta"])
direccion_ip = st.text_input("Dirección IP")
hora_transaccion = st.number_input("Hora de la transacción")

# Botón para analizar la transacción
if st.button("Analizar Transacción"):
    if not validar_direccion_ip(direccion_ip):
        st.error("Ingrese una dirección IP válida.")
    else:
        datos_transaccion = {
            "monto": monto,
            "ubicacion": ubicacion,
            "usuario": usuario,
            "metodo_pago": metodo_pago,
            "dispositivo": dispositivo,
            "direccion_ip": direccion_ip,
            "hora_transaccion": hora_transaccion
        }

        alerta_openai = analizar_transaccion_openai(datos_transaccion)
        st.write("Análisis de OpenAI:", alerta_openai)

        prediccion_ml = predecir_fraude_ml(modelo_ml, datos_transaccion)
        st.write("Predicción de Machine Learning:", "Fraude" if prediccion_ml == 1 else "No fraude")