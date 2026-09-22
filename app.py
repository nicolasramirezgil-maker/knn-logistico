import streamlit as st
import numpy as np
import joblib

knn = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler_kmeans.joblib')

PERFILES = {
    0: ("Cluster 0 - Paquete ligero de alto valor con friccion",
        "Producto costoso y liviano con alta interaccion de servicio al cliente. Requiere atencion prioritaria de soporte y seguimiento personalizado."),
    1: ("Cluster 1 - Operacion estandar",
        "Paquete pesado con bajo descuento y poca friccion. Flujo logistico normal sin intervencion especial requerida."),
    2: ("Cluster 2 - Envio de alto riesgo por descuento agresivo",
        "Descuento superior al 10%, paquete liviano y producto de bajo costo. Alta probabilidad de retraso. Se recomienda refuerzo de capacidad logistica preventiva.")
}

st.set_page_config(page_title="Clasificador Logistico", page_icon="=", layout="centered")
st.title("Clasificador de Segmento Logistico")
st.markdown("Ingresa los parametros del envio para obtener su segmento operativo.")
st.divider()
col1, col2 = st.columns(2)
with col1:
    discount = st.slider("Descuento ofrecido (%)", min_value=1, max_value=65, value=7)
    weight = st.slider("Peso del paquete (g)", min_value=1000, max_value=8000, value=3500, step=50)
with col2:
    calls = st.slider("Llamadas al servicio al cliente", min_value=2, max_value=7, value=4)
    cost = st.slider("Costo del producto (USD)", min_value=96, max_value=310, value=200)
st.divider()
if st.button("Clasificar envio", use_container_width=True):
    X_new = np.array([[discount, weight, calls, cost]])
    X_scaled = scaler.transform(X_new)
    cluster = knn.predict(X_scaled)[0]
    titulo, descripcion = PERFILES[cluster]
    st.subheader(titulo)
    st.info(descripcion)
    st.markdown(f"**Variables ingresadas:** Descuento={discount}% | Peso={weight}g | Llamadas={calls} | Costo=${cost}")
