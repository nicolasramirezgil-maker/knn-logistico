import streamlit as st
import numpy as np
import joblib

knn    = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler_kmeans.joblib')

PERFILES = {
    0: {
        "titulo":      "Cluster 0 — Paquete de alto valor con fricción",
        "descripcion": "Producto costoso y liviano con alta interacción de servicio al cliente. Requiere atención prioritaria de soporte y seguimiento personalizado.",
        "color":  "#D97706",
        "bg":     "#FFFBEB",
        "border": "#F59E0B",
        "badge":  "⚠️ ATENCIÓN PRIORITARIA",
        "icon":   "📦",
        "tip":    "💡 Asignar agente de soporte dedicado y activar seguimiento proactivo del envío.",
        "fn":     "warning",
    },
    1: {
        "titulo":      "Cluster 1 — Operación estándar",
        "descripcion": "Paquete pesado con bajo descuento y poca fricción. Flujo logístico normal sin intervención especial requerida.",
        "color":  "#059669",
        "bg":     "#ECFDF5",
        "border": "#10B981",
        "badge":  "✅ FLUJO NORMAL",
        "icon":   "📗",
        "tip":    "💡 Procesar en flujo estándar. Sin intervención especial requerida.",
        "fn":     "success",
    },
    2: {
        "titulo":      "Cluster 2 — Envío de alto riesgo",
        "descripcion": "Descuento superior al 10%, paquete liviano y producto de bajo costo. Alta probabilidad de retraso. Se recomienda refuerzo de capacidad logística preventiva.",
        "color":  "#DC2626",
        "bg":     "#FEF2F2",
        "border": "#EF4444",
        "badge":  "🚨 ALTO RIESGO",
        "icon":   "⛔",
        "tip":    "💡 Revisar capacidad logística disponible. Considerar limitar descuentos en zonas de baja cobertura.",
        "fn":     "error",
    },
}

st.set_page_config(page_title="Clasificador Logístico", page_icon="📦", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #F1F5F9;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 760px;
}
h1, h2, h3 { color: #0F172A !important; }
label[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 0.92rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.025em !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
}
[data-testid="metric-container"] {
    background: white;
    border-radius: 10px;
    padding: 0.8rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F0;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #1E3A5F !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
hr { border-color: #CBD5E1 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📦 Clasificador de Segmento Logístico")
st.markdown(
    '<p style="color:#64748B;font-size:1rem;margin-top:-0.6rem;margin-bottom:1.4rem;">'
    'Ingresa los parámetros del envío para obtener el segmento operativo del pedido.'
    '</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    discount = st.slider("💸  Descuento ofrecido (%)", min_value=1,    max_value=65,   value=7)
    weight   = st.slider("⚖️  Peso del paquete (g)",   min_value=1000, max_value=8000, value=3500, step=50)
with col2:
    calls = st.slider("📞  Llamadas al cliente",      min_value=2,  max_value=7,   value=4)
    cost  = st.slider("💰  Costo del producto (USD)", min_value=96, max_value=310, value=200)

st.markdown("---")

st.markdown('<p style="font-size:0.78rem;font-weight:700;color:#94A3B8;letter-spacing:0.08em;margin-bottom:0.6rem;">RESUMEN DE VARIABLES</p>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Descuento", f"{discount}%")
m2.metric("Peso",      f"{weight:,} g")
m3.metric("Llamadas",  calls)
m4.metric("Costo",     f"${cost}")

st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)

if st.button("🔍  Clasificar envío", use_container_width=True):
    X_scaled = scaler.transform(np.array([[discount, weight, calls, cost]]))
    cluster  = knn.predict(X_scaled)[0]
    p        = PERFILES[cluster]

    st.markdown(f"""
    <div style="
        background: {p['bg']};
        border: 2px solid {p['border']};
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 16px {p['border']}30;
    ">
        <span style="
            background: {p['border']}25;
            color: {p['color']};
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            padding: 0.28rem 0.75rem;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 0.75rem;
        ">{p['badge']}</span>
        <div style="
            font-size: 1.15rem;
            font-weight: 700;
            color: {p['color']};
            margin-bottom: 0.5rem;
        ">{p['icon']}  {p['titulo']}</div>
        <div style="color: #374151; font-size: 0.95rem; line-height: 1.65;">
            {p['descripcion']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.9rem'></div>", unsafe_allow_html=True)
    fn_map = {"warning": st.warning, "success": st.success, "error": st.error}
    fn_map[p["fn"]](p["tip"])
