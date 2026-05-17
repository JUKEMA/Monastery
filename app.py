"""
app.py · MONASTERY Club — Analítica Predictiva de Aforo y Demanda
Streamlit entry point — navegación multi-página con sidebar institucional
Minería de Datos 801 · UDeC · 2026
"""

import streamlit as st

st.set_page_config(
    page_title="MONASTERY Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global institucional (dark theme MONASTERY)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: #0A0A0F !important;
    border-right: 1px solid #1A1A2E;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stSidebar"] hr { border-color: #1A1A2E !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #C9A227 !important;
    font-size: 0.72rem !important;
    letter-spacing: .09em;
    text-transform: uppercase;
    font-weight: 600;
}

[data-testid="stAppViewContainer"] > .main { background: #0D0D14; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px; }

[data-testid="metric-container"] {
    background: #12121C;
    border: 1px solid #1E1E30;
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.74rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600;
    color: #E6E0C8 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #1E1E30 !important;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stTabs"] [role="tablist"] {
    background: #12121C;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #1E1E30;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #8B949E !important;
    font-size: 0.82rem !important;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #C9A227 !important;
    border-bottom: 2px solid #C9A227 !important;
    background: transparent !important;
}
.stButton > button[kind="primary"] {
    background: #C9A227 !important;
    border: none !important;
    color: #0A0A0F !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"]:hover { background: #E8B830 !important; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #12121C !important;
    border: 1px solid #2A2A3E !important;
    color: #E6E0C8 !important;
    border-radius: 6px !important;
}
details summary {
    background: #12121C;
    border: 1px solid #1E1E30;
    border-radius: 8px;
    padding: .5rem 1rem;
    color: #C9D1D9 !important;
    font-weight: 500;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D0D14; }
::-webkit-scrollbar-thumb { background: #2A2A3E; border-radius: 3px; }

.page-header {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1E1E30;
    margin-bottom: 1.5rem;
}
.page-header h1 { font-size: 1.35rem; font-weight: 600; color: #E6E0C8; margin: 0; }
.page-header .subtitle { font-size: 0.8rem; color: #8B949E; margin: 0; }

.risk-card {
    background: #12121C;
    border: 1px solid #1E1E30;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.alert-critico {
    background: rgba(239,68,68,.12);
    border-left: 4px solid #EF4444;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #FCA5A5;
    font-size: .88rem;
    margin: .5rem 0;
}
.alert-moderado {
    background: rgba(201,162,39,.12);
    border-left: 4px solid #C9A227;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #F0D080;
    font-size: .88rem;
    margin: .5rem 0;
}
.alert-normal {
    background: rgba(63,185,80,.12);
    border-left: 4px solid #3FB950;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #6EE7B7;
    font-size: .88rem;
    margin: .5rem 0;
}
.info-box {
    background: rgba(201,162,39,.08);
    border: 1px solid rgba(201,162,39,.3);
    border-radius: 8px;
    padding: .75rem 1rem;
    color: #E8C84A;
    font-size: .85rem;
    margin: .5rem 0;
}
.warn-box {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.25);
    border-radius: 8px;
    padding: .75rem 1rem;
    color: #FCA5A5;
    font-size: .85rem;
    margin: .5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Estado global
if "modelo" not in st.session_state:
    from modelo_monastery import ModeloMonastery
    st.session_state.modelo = ModeloMonastery()
if "df_master" not in st.session_state:
    st.session_state.df_master = None

# ── Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 .5rem">
      <div style="font-size:1.5rem;font-weight:700;color:#C9A227;letter-spacing:-.01em">
        🎵 MONASTERY
      </div>
      <div style="font-size:.7rem;color:#6E7681;margin-top:.15rem;font-weight:500;letter-spacing:.04em">
        ANALYTICS · CLUB FACATATIVÁ
      </div>
      <div style="font-size:.68rem;color:#2A2A3E;margin-top:.3rem">
        Minería de Datos 801 · UDeC · 2026
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### Módulos")
    st.page_link("app.py",                       label="🏠  Inicio")
    st.page_link("pages/01_entrenamiento.py",     label="⚡  Entrenamiento")
    st.page_link("pages/02_dashboard_eda.py",     label="📊  Dashboard & EDA")
    st.page_link("pages/03_gap_analysis.py",      label="📉  Análisis de Brechas")
    st.page_link("pages/04_prescripcion.py",      label="🎯  Prescripción")
    st.page_link("pages/05_simulacion.py",        label="🔮  Simulación de Escalamiento")
    st.page_link("pages/06_validacion.py",        label="✅  Validación & Confianza")

    st.divider()
    modelo = st.session_state.modelo
    if modelo.trained:
        rf  = modelo.resultados_clf.get("Random Forest Clf.", {})
        rfr = modelo.resultados_reg.get("Random Forest Reg.", {})
        st.markdown(f"""
        <div style="background:#0A1A08;border:1px solid #1A4020;border-radius:8px;padding:.75rem 1rem">
          <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.05em">
            ✓ Modelo activo
          </div>
          <div style="font-size:.82rem;color:#7EE2A8;margin:.3rem 0 .1rem">Random Forest</div>
          <div style="font-size:.73rem;color:#3FB950;line-height:1.7">
            Clf F1: {rf.get('f1','—')} · AUC: {rf.get('roc_auc','—')}<br>
            Reg R²: {rfr.get('r2','—')} · RMSE: {rfr.get('rmse','—')}<br>
            Train: {modelo.n_train} · Test: {modelo.n_test}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1A0A08;border:1px solid #4D2000;border-radius:8px;padding:.75rem 1rem">
          <div style="font-size:.7rem;color:#C9A227;font-weight:600">⚠ Sin modelo entrenado</div>
          <div style="font-size:.73rem;color:#6B4A10;margin-top:.3rem">
            Carga el Excel en ⚡ Entrenamiento y entrena el modelo.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:.68rem;color:#3A3A52;line-height:1.7">
      <b style="color:#4A4A62">Equipo</b><br>
      Maicol López · Daniela Romero<br>
      Juan Pérez · Carlos Torrado<br><br>
      <b style="color:#4A4A62">Docente</b><br>
      Oscar Jobany Gómez Ochoa<br><br>
      <b style="color:#4A4A62">Asignatura</b><br>
      Minería de Datos — 801
    </div>
    """, unsafe_allow_html=True)

# ── Página de inicio
st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🎵</span>
  <div>
    <h1>MONASTERY Club — Analítica Predictiva de Aforo y Demanda</h1>
    <p class="subtitle">Modelo predictivo y prescriptivo · CRISP-DM · Minería de Datos 801 · Universidad de Cundinamarca · 2026</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Tarjetas de flujo
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        ⚡ Flujo de trabajo
      </div>
      <div style="font-size:.86rem;color:#C9D1D9;line-height:2">
        1 · Cargar dataset Excel (9 tablas)<br>
        2 · Entrenar modelos reg. + clasificación<br>
        3 · Explorar Dashboard EDA<br>
        4 · Analizar brechas de aforo<br>
        5 · Obtener prescripciones por evento<br>
        6 · Simular escalamiento de capacidad<br>
        7 · Validar predicciones históricas
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        📦 Dataset MONASTERY
      </div>
      <div style="font-size:.86rem;color:#C9D1D9;line-height:2">
        <span style="color:#8B949E">Eventos:</span> 150 registros históricos<br>
        <span style="color:#8B949E">Tablas:</span> 9 (esquema estrella)<br>
        <span style="color:#8B949E">Features modelo:</span> 11 variables<br>
        <span style="color:#8B949E">Archivo:</span> dataset_eventos_v3.xlsx<br>
        <span style="color:#8B949E">Período:</span> 2024 completo
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#EF4444;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        🎯 Variables objetivo
      </div>
      <div style="font-size:.86rem;color:#C9D1D9;line-height:2.1">
        <span style="color:#EF4444">■ Aforo Crítico</span> → Ocupación ≥ 85%<br>
        <span style="color:#3FB950">■ Aforo Normal</span> &nbsp;→ Ocupación &lt; 85%<br>
        <span style="color:#C9A227">◆ Regresión</span>&nbsp;&nbsp;&nbsp;&nbsp;→ Asistencia total<br>
        <span style="color:#8B949E">Modelo reg.:</span> Random Forest / XGBoost<br>
        <span style="color:#8B949E">Modelo clf.:</span> RF Clasificador · F1 · AUC
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# CRISP-DM phases
phases = [
    ("1", "Comprensión\ndel negocio",   "#C9A227", "Optimizar staffing y eventos con datos históricos."),
    ("2", "Comprensión\nde los datos",  "#3FB950", "150 eventos · 9 tablas · clima, festivos, preventa."),
    ("3", "Preparación\nde datos",      "#58A6FF", "Merge esquema estrella · encoding · KPIs derivados."),
    ("4", "Modelado",                   "#8B5CF6", "RF + XGBoost · Regresión + Clasificación binaria."),
    ("5", "Evaluación",                 "#EF4444", "RMSE · R² · F1 · AUC-ROC · CV-5 estratificado."),
    ("6", "Comunicación",               "#10B981", "Dashboard prescriptivo · Simulación · Validación."),
]
cols = st.columns(6)
for col, (num, label, color, desc) in zip(cols, phases):
    with col:
        st.markdown(f"""
        <div style="background:#12121C;border:1px solid #1E1E30;border-top:3px solid {color};
                    border-radius:8px;padding:.85rem .75rem;text-align:center;height:135px;
                    display:flex;flex-direction:column;justify-content:center;">
          <div style="font-size:1.35rem;font-weight:700;color:{color}">{num}</div>
          <div style="font-size:.76rem;font-weight:600;color:#E6E0C8;margin:.2rem 0;
                      white-space:pre-line;line-height:1.3">{label}</div>
          <div style="font-size:.67rem;color:#8B949E;line-height:1.4">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
  🎵 &nbsp;<b>Proyecto:</b> Modelo de analítica predictiva de aforo y demanda para la optimización del personal operativo
  y la planeación estratégica de eventos en la discoteca MONASTERY Club en Facatativá.
  El sistema predice si un evento alcanzará aforo crítico (&ge;85%) y prescribe acciones operativas concretas
  con hasta 48 horas de anticipación.
</div>
""", unsafe_allow_html=True)
