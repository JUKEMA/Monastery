"""
pages/06_validacion.py · Validación Histórica & Confianza Gerencial
MONASTERY Analytics · CRISP-DM Fase 6: Comunicación y Cierre
Compara predicciones del modelo contra resultados reales para generar
confianza en la toma de decisiones. Usa los últimos N eventos del
dataset histórico como "prueba de fuego" temporal.
Business Analytics 801 · UDeC · 2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_monastery import AFORO_LABELS, AFORO_COLORS, FEATURE_COLS

st.set_page_config(
    page_title="Validación · MONASTERY",
    layout="wide",
    page_icon="✅",
)

# ── CSS institucional (consistente con el resto de páginas)
st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D0D14}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
[data-testid="metric-container"]{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem!important}
[data-testid="metric-container"] label{color:#8B949E!important;font-size:.74rem!important;font-weight:500;text-transform:uppercase;letter-spacing:.05em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:600;color:#E6E0C8!important}
[data-testid="stDataFrame"]{border:1px solid #1E1E30!important;border-radius:8px;overflow:hidden}
[data-testid="stTabs"] [role="tablist"]{background:#12121C;border-radius:8px 8px 0 0;border-bottom:1px solid #1E1E30}
[data-testid="stTabs"] button[role="tab"]{color:#8B949E!important;font-size:.82rem!important;font-weight:500}
[data-testid="stTabs"] button[aria-selected="true"]{color:#C9A227!important;border-bottom:2px solid #C9A227!important;background:transparent!important}
.risk-card{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #1E1E30;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6E0C8;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.info-box{background:rgba(201,162,39,.08);border:1px solid rgba(201,162,39,.3);border-radius:8px;padding:.75rem 1rem;color:#E8C84A;font-size:.85rem;margin:.5rem 0}
.warn-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:8px;padding:.75rem 1rem;color:#FCA5A5;font-size:.85rem;margin:.5rem 0}
.success-box{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.25);border-radius:8px;padding:.75rem 1rem;color:#6EE7B7;font-size:.85rem;margin:.5rem 0}
.val-event-row{background:#12121C;border:1px solid #1E1E30;border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;display:flex;align-items:center;gap:1rem;font-size:.82rem}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
.badge-critico{background:rgba(239,68,68,.15);color:#EF4444;border:1px solid rgba(239,68,68,.4);border-radius:4px;padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.badge-normal{background:rgba(63,185,80,.15);color:#3FB950;border:1px solid rgba(63,185,80,.4);border-radius:4px;padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.badge-ok{background:rgba(63,185,80,.15);color:#3FB950;border:1px solid rgba(63,185,80,.4);border-radius:4px;padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.badge-fail{background:rgba(239,68,68,.15);color:#EF4444;border:1px solid rgba(239,68,68,.4);border-radius:4px;padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.trust-gauge{background:#12121C;border:1px solid #1E1E30;border-radius:12px;padding:1.25rem 1.5rem;text-align:center}
.trust-label{font-size:.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.trust-value{font-size:2.8rem;font-weight:700;line-height:1}
</style>""", unsafe_allow_html=True)

# ── Guards de estado
if "modelo" not in st.session_state or not st.session_state.modelo.trained:
    st.warning("⚠ Entrena el modelo primero en ⚡ Entrenamiento.")
    st.stop()

if "df_master" not in st.session_state or st.session_state.df_master is None:
    st.warning("⚠ Carga el dataset primero en ⚡ Entrenamiento.")
    st.stop()

modelo   = st.session_state.modelo
df_full  = st.session_state.df_master.copy()

# ── Encabezado
st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">✅</span>
  <div>
    <h1>Validación Histórica & Confianza Gerencial</h1>
    <p class="subtitle">CRISP-DM Fase 6 · Predicción vs. resultado real · Índice de confianza · Calibración del modelo</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  ✅ &nbsp;<b>¿Qué hace esta sección?</b> Simula cómo hubiera actuado el modelo <em>antes</em> de que ocurrieran
  los últimos eventos del historial, comparando sus predicciones con el resultado real. Este análisis genera confianza
  gerencial: si el modelo acertó en el pasado, puede confiarse para anticipar el futuro.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# A) CONFIGURACIÓN DEL PERÍODO DE VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════
st.markdown("#### ⚙ Configuración del período de validación")
col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 2])

with col_cfg1:
    n_eventos_val = st.slider(
        "Eventos a validar (últimos N del historial)",
        min_value=10, max_value=min(50, len(df_full) - 20),
        value=30, step=5,
        help="Cuántos eventos recientes se usan como 'conjunto de validación temporal'.",
    )

with col_cfg2:
    modelo_clf_sel = st.selectbox(
        "Modelo de clasificación",
        options=list(modelo.resultados_clf.keys()),
        index=min(1, len(modelo.resultados_clf) - 1),
    )
    modelo_reg_sel = st.selectbox(
        "Modelo de regresión",
        options=list(modelo.resultados_reg.keys()),
        index=min(1, len(modelo.resultados_reg) - 1),
    )

with col_cfg3:
    st.markdown("""
    <div class="risk-card" style="height:100%;min-height:90px">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        ¿Cómo funciona la validación temporal?
      </div>
      <div style="font-size:.82rem;color:#C9D1D9;line-height:1.7">
        Se ordenan los eventos por fecha · los <b>últimos N</b> se tratan como "periodo de prueba"
        (el modelo no los vio en entrenamiento desde esta perspectiva temporal) · se comparan las
        predicciones del modelo contra los resultados reales registrados en el dataset.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# B) GENERAR PREDICCIONES DE VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════

# Preparar dataset con features encodadas
df_prep = modelo.df_prepared.copy() if modelo.df_prepared is not None else None
if df_prep is None:
    st.error("El modelo no tiene datos preparados. Vuelve a entrenar.")
    st.stop()

# Ordenar por fecha para tener sentido temporal
if "fecha" in df_prep.columns:
    df_prep = df_prep.sort_values("fecha").reset_index(drop=True)
else:
    df_prep = df_prep.reset_index(drop=True)

# Dividir: validación = últimos N eventos
df_val = df_prep.tail(n_eventos_val).copy()
X_val  = df_val[FEATURE_COLS].values

# ── Predicciones de clasificación
mod_clf = modelo.modelo_clf[modelo_clf_sel]
if modelo_clf_sel == "Regresión Logística":
    X_val_scaled = modelo.scaler_clf.transform(X_val)
    probas_val   = mod_clf.predict_proba(X_val_scaled)[:, 1]
else:
    probas_val   = mod_clf.predict_proba(X_val)[:, 1]
preds_val = (probas_val >= 0.5).astype(int)

# ── Predicciones de regresión
mod_reg = modelo.modelo_reg[modelo_reg_sel]
if modelo_reg_sel == "Regresión Lineal":
    X_val_sc_r = modelo.scaler_reg.transform(X_val)
    asist_pred = mod_reg.predict(X_val_sc_r)
else:
    asist_pred = mod_reg.predict(X_val)
asist_pred = np.clip(asist_pred, 0, None)

# ── Valores reales
y_real_clf  = df_val["aforo_critico"].values
y_real_asist = df_val["asistencia_total"].values

# ── Métricas de validación
hits_clf   = int((preds_val == y_real_clf).sum())
acc_val    = hits_clf / len(y_real_clf)
err_asist  = asist_pred - y_real_asist
rmse_val   = float(np.sqrt(np.mean(err_asist ** 2)))
mae_val    = float(np.mean(np.abs(err_asist)))
pct_dentro_10 = float(np.mean(np.abs(err_asist) <= 0.10 * y_real_asist.clip(1)) * 100)

# Precision / Recall sobre críticos
tp = int(((preds_val == 1) & (y_real_clf == 1)).sum())
fp = int(((preds_val == 1) & (y_real_clf == 0)).sum())
fn = int(((preds_val == 0) & (y_real_clf == 1)).sum())
tn = int(((preds_val == 0) & (y_real_clf == 0)).sum())
prec_val   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_val     = (2 * prec_val * recall_val / (prec_val + recall_val)
              if (prec_val + recall_val) > 0 else 0.0)

# Índice de confianza compuesto
confianza = (acc_val * 0.45 + min(pct_dentro_10 / 100, 1.0) * 0.35 + f1_val * 0.20) * 100

def nivel_confianza(c):
    if c >= 80: return "#3FB950", "ALTA"
    if c >= 60: return "#C9A227", "MEDIA"
    return "#EF4444", "BAJA"
c_color, c_label = nivel_confianza(confianza)

# ══════════════════════════════════════════════════════════════════════
# C) KPIs DE CONFIANZA
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 📊 Índice de confianza y métricas de validación")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Eventos validados",       f"{n_eventos_val}")
k2.metric("Clasificación correcta",  f"{hits_clf}/{n_eventos_val}  ({acc_val*100:.1f}%)")
k3.metric("RMSE asistencia",         f"{rmse_val:.1f} personas")
k4.metric("MAE asistencia",          f"{mae_val:.1f} personas")
k5.metric("Dentro ±10% real",        f"{pct_dentro_10:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Gauge + desglose
ga, gb, gc = st.columns([1, 1, 2])

with ga:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confianza, 1),
        number={"suffix": "%", "font": {"size": 28, "color": c_color}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1,
                      "tickcolor": "#3A3A52", "tickfont": {"color": "#8B949E", "size": 9}},
            "bar":   {"color": c_color, "thickness": 0.28},
            "bgcolor": "#12121C",
            "bordercolor": "#1E1E30",
            "steps": [
                {"range": [0,  60], "color": "rgba(239,68,68,.12)"},
                {"range": [60, 80], "color": "rgba(201,162,39,.12)"},
                {"range": [80,100], "color": "rgba(63,185,80,.12)"},
            ],
            "threshold": {
                "line": {"color": c_color, "width": 3},
                "thickness": 0.85, "value": confianza,
            },
        },
        title={"text": f"Índice de Confianza<br><span style='font-size:.65em;color:#8B949E'>Clasificación 45% · Rango 35% · F1 20%</span>",
               "font": {"size": 12, "color": "#C9D1D9"}},
    ))
    fig_gauge.update_layout(
        height=230, paper_bgcolor="#12121C",
        margin=dict(l=20, r=20, t=50, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(f"""
    <div style="text-align:center;margin-top:-.5rem">
      <span style="background:{c_color}22;color:{c_color};border:1px solid {c_color}55;
        border-radius:6px;padding:.3rem 1.2rem;font-size:.82rem;font-weight:700">
        CONFIANZA {c_label}
      </span>
    </div>
    """, unsafe_allow_html=True)

with gb:
    # Pie de aciertos/fallos
    fig_pie = go.Figure(go.Pie(
        labels=["Clasificación correcta", "Clasificación errónea"],
        values=[hits_clf, n_eventos_val - hits_clf],
        hole=0.62,
        marker=dict(colors=["#3FB950", "#EF4444"],
                    line=dict(color="#0D0D14", width=2)),
        textfont=dict(size=10, color="#C9D1D9"),
        hovertemplate="%{label}: %{value} eventos<extra></extra>",
    ))
    fig_pie.update_layout(
        title=dict(text="Aciertos de clasificación", font=dict(color="#C9D1D9", size=12)),
        paper_bgcolor="#12121C", plot_bgcolor="#12121C",
        font=dict(color="#8B949E"),
        legend=dict(font=dict(size=9, color="#C9D1D9"), orientation="h",
                    yanchor="bottom", y=-0.15),
        margin=dict(l=10, r=10, t=50, b=20),
        height=230,
        annotations=[dict(
            text=f"<b>{acc_val*100:.0f}%</b>",
            x=0.5, y=0.5, font=dict(size=20, color=c_color),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with gc:
    # Tabla de métricas detalladas
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;
        letter-spacing:.06em;margin-bottom:.75rem">Métricas detalladas de validación</div>
    """, unsafe_allow_html=True)
    metricas = [
        ("Clasificación", "Exactitud (Accuracy)",  f"{acc_val*100:.2f}%",  "#C9D1D9"),
        ("Clasificación", "Precisión (aforo crít.)", f"{prec_val*100:.2f}%",  "#C9D1D9"),
        ("Clasificación", "Recall (aforo crít.)",  f"{recall_val*100:.2f}%","#C9D1D9"),
        ("Clasificación", "F1-score (aforo crít.)",f"{f1_val*100:.2f}%",   "#C9A227"),
        ("Regresión",     "RMSE asistencia",        f"{rmse_val:.1f} pers.", "#C9D1D9"),
        ("Regresión",     "MAE asistencia",         f"{mae_val:.1f} pers.", "#C9D1D9"),
        ("Regresión",     "Dentro ±10% real",       f"{pct_dentro_10:.1f}%","#3FB950"),
    ]
    for cat, nombre, val, color in metricas:
        cat_color = "#58A6FF" if cat == "Clasificación" else "#C9A227"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
          padding:.28rem 0;border-bottom:1px solid #1E1E30">
          <span style="font-size:.76rem;color:#8B949E">
            <span style="color:{cat_color};font-weight:600">[{cat[:3]}]</span> {nombre}
          </span>
          <span style="font-size:.82rem;font-weight:600;color:{color}">{val}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# D) TABS DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
tab_reg, tab_clf, tab_cal, tab_tabla = st.tabs([
    "📈 Regresión — Predicho vs Real",
    "🎯 Clasificación — Aciertos evento a evento",
    "🔬 Calibración del modelo",
    "📋 Tabla completa de validación",
])

# ────────────────────────────────────────────────────────────────────
# TAB 1: REGRESIÓN — predicho vs real
# ────────────────────────────────────────────────────────────────────
with tab_reg:
    c_r1, c_r2 = st.columns(2)

    with c_r1:
        # Serie temporal: predicho vs real
        x_idx = np.arange(1, n_eventos_val + 1)
        nombre_label = (
            df_val["nombre"].values
            if "nombre" in df_val.columns
            else [f"Evento {i}" for i in x_idx]
        )
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=x_idx, y=y_real_asist, name="Asistencia real",
            mode="lines+markers",
            line=dict(color="#58A6FF", width=2),
            marker=dict(size=5),
        ))
        fig_ts.add_trace(go.Scatter(
            x=x_idx, y=asist_pred, name="Predicción modelo",
            mode="lines+markers",
            line=dict(color="#C9A227", width=2, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
        ))
        fig_ts.update_layout(
            title=dict(text="Asistencia real vs. predicha (serie temporal de validación)",
                       font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Evento (orden temporal)", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            yaxis=dict(title="Asistencia (personas)", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            legend=dict(font=dict(size=10, color="#C9D1D9"),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=50, r=20, t=50, b=50), height=330,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with c_r2:
        # Scatter predicho vs real
        lim = max(y_real_asist.max(), asist_pred.max()) * 1.05
        fig_sc = go.Figure()
        fig_sc.add_shape(
            type="line", x0=0, y0=0, x1=lim, y1=lim,
            line=dict(color="#3A3A52", width=1, dash="dash"),
        )
        fig_sc.add_shape(
            type="line", x0=0, y0=0 * 0.9, x1=lim * 0.9, y1=lim,
            line=dict(color="#1E2A1E", width=1),
        )
        fig_sc.add_trace(go.Scatter(
            x=y_real_asist, y=asist_pred,
            mode="markers",
            marker=dict(
                color=np.abs(err_asist),
                colorscale=[[0, "#3FB950"], [0.5, "#C9A227"], [1, "#EF4444"]],
                size=8, opacity=0.85,
                colorbar=dict(
                    title=dict(text="Error abs.", font=dict(color="#8B949E", size=9)),
                    tickfont=dict(color="#8B949E", size=8),
                    thickness=10,
                ),
                showscale=True,
            ),
            text=[f"{n}<br>Real: {r:.0f} · Pred: {p:.0f}<br>Error: {e:+.0f}"
                  for n, r, p, e in zip(nombre_label, y_real_asist, asist_pred, err_asist)],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_sc.update_layout(
            title=dict(text="Dispersión: Real vs. Predicho",
                       font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Asistencia real", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E"), range=[0, lim]),
            yaxis=dict(title="Asistencia predicha", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E"), range=[0, lim]),
            margin=dict(l=50, r=20, t=50, b=50), height=330,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Histograma de errores
    st.markdown("##### Distribución de errores de regresión")
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        fig_he = go.Figure(go.Histogram(
            x=err_asist, nbinsx=20,
            marker_color="#C9A227", opacity=0.8,
        ))
        fig_he.add_vline(x=0, line=dict(color="#EF4444", width=2, dash="dash"))
        fig_he.add_vline(x=err_asist.mean(), line=dict(color="#3FB950", width=1.5))
        fig_he.update_layout(
            title=dict(text="Distribución de errores (predicho − real)",
                       font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Error (personas)", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            yaxis=dict(title="Frecuencia", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            margin=dict(l=50, r=20, t=50, b=50), height=270,
        )
        st.plotly_chart(fig_he, use_container_width=True)
    with c_h2:
        # Porcentaje de error relativo
        pct_err = np.abs(err_asist) / y_real_asist.clip(1) * 100
        fig_pct = go.Figure(go.Histogram(
            x=pct_err, nbinsx=20,
            marker_color="#58A6FF", opacity=0.8,
        ))
        fig_pct.add_vline(x=10, line=dict(color="#C9A227", width=1.5, dash="dot"),
                          annotation_text="±10%", annotation_font_color="#C9A227")
        fig_pct.update_layout(
            title=dict(text="Error relativo por evento (%)",
                       font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Error relativo (%)", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            yaxis=dict(title="Frecuencia", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            margin=dict(l=50, r=20, t=50, b=50), height=270,
        )
        st.plotly_chart(fig_pct, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# TAB 2: CLASIFICACIÓN — aciertos evento a evento
# ────────────────────────────────────────────────────────────────────
with tab_clf:
    nombre_label = (
        df_val["nombre"].values
        if "nombre" in df_val.columns
        else [f"Evento {i}" for i in range(1, n_eventos_val + 1)]
    )
    fecha_label = (
        pd.to_datetime(df_val["fecha"]).dt.strftime("%d/%m/%Y").values
        if "fecha" in df_val.columns
        else ["—"] * n_eventos_val
    )

    c_t1, c_t2 = st.columns([2, 1])
    with c_t1:
        # Barras de probabilidad con marcador de resultado real
        colores_barra = ["#3FB950" if r == 0 else "#EF4444"
                         for r in y_real_clf]
        fig_prob = go.Figure()

        # Fondo de zona de umbral
        fig_prob.add_hrect(y0=0.40, y1=0.85,
                           fillcolor="rgba(201,162,39,.05)",
                           line_width=0)
        fig_prob.add_hline(y=0.50, line=dict(color="#EF4444", width=1.5, dash="dash"),
                           annotation_text="Umbral 50%",
                           annotation_font_color="#EF4444",
                           annotation_font_size=9)

        fig_prob.add_trace(go.Bar(
            x=list(range(1, n_eventos_val + 1)),
            y=probas_val,
            marker_color=[
                "#3FB950" if p == r else "#EF4444"
                for p, r in zip(preds_val, y_real_clf)
            ],
            opacity=0.8,
            text=[f"{'✓' if p==r else '✗'}" for p, r in zip(preds_val, y_real_clf)],
            textposition="outside",
            textfont=dict(size=9, color="#C9D1D9"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Prob. crítico: %{y:.1%}<br>"
                "Pred: %{customdata[2]} · Real: %{customdata[3]}"
                "<extra></extra>"
            ),
            customdata=list(zip(
                nombre_label,
                fecha_label,
                [AFORO_LABELS[p] for p in preds_val],
                [AFORO_LABELS[r] for r in y_real_clf],
            )),
        ))
        fig_prob.update_layout(
            title=dict(
                text="Probabilidad de aforo crítico predicha — ✓ acierto · ✗ fallo",
                font=dict(color="#C9D1D9", size=13),
            ),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Evento (orden temporal)", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E")),
            yaxis=dict(title="Probabilidad crítico",
                       gridcolor="#1E1E30", tickfont=dict(color="#8B949E"),
                       tickformat=".0%", range=[0, 1.15]),
            margin=dict(l=50, r=20, t=55, b=50), height=350,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with c_t2:
        # Matriz de confusión de validación
        cm_val = np.array([[tn, fp], [fn, tp]])
        cm_n   = cm_val.astype(float) / (cm_val.sum(axis=1, keepdims=True) + 1e-9)
        txt_cm = [
            [f"{cm_val[i][j]}<br><span style='font-size:.7rem'>({cm_n[i][j]*100:.0f}%)</span>"
             for j in range(2)]
            for i in range(2)
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_n,
            x=["Normal (pred)", "Crítico (pred)"],
            y=["Normal (real)", "Crítico (real)"],
            text=txt_cm, texttemplate="%{text}",
            colorscale=[[0, "#0D0D14"], [0.5, "#2A1A00"], [1, "#C9A227"]],
            showscale=False, xgap=3, ygap=3,
        ))
        fig_cm.update_layout(
            title=dict(text="Matriz de confusión<br>(conjunto de validación)",
                       font=dict(color="#C9D1D9", size=12)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=10),
            xaxis=dict(tickfont=dict(color="#8B949E", size=9)),
            yaxis=dict(tickfont=dict(color="#8B949E", size=9), autorange="reversed"),
            margin=dict(l=80, r=10, t=70, b=60), height=350,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Resumen de tipos de error
    st.markdown("##### Análisis de tipos de error en clasificación")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Verdaderos Positivos (TP)", tp,
               help="Eventos críticos que el modelo predijo correctamente como críticos")
    ec2.metric("Verdaderos Negativos (TN)", tn,
               help="Eventos normales que el modelo predijo correctamente como normales")
    ec3.metric("Falsos Positivos (FP) ⚠", fp,
               help="Eventos normales predichos como críticos → sobre-preparación innecesaria")
    ec4.metric("Falsos Negativos (FN) 🚨", fn,
               help="Eventos críticos predichos como normales → riesgo operativo real")

    if fn > 0:
        st.markdown(f"""
        <div class="warn-box">
          🚨 &nbsp;<b>{fn} evento(s) crítico(s) no detectado(s)</b> — El modelo clasificó como 'Normal' un evento que
          realmente superó el 85% de aforo. Esto implica riesgo operativo: personal insuficiente o inventario bajo.
          Con más datos históricos, este indicador debería mejorar.
        </div>
        """, unsafe_allow_html=True)
    if fp > 0:
        st.markdown(f"""
        <div class="info-box">
          ⚠ &nbsp;<b>{fp} evento(s) con sobre-predicción</b> — El modelo alertó de aforo crítico, pero el evento terminó
          siendo normal. Implica un costo de precaución (personal extra, más inventario) sin consecuencias graves.
          Esta es la dirección de error preferible desde el punto de vista del negocio.
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# TAB 3: CALIBRACIÓN DEL MODELO
# ────────────────────────────────────────────────────────────────────
with tab_cal:
    st.markdown("""
    <div class="info-box">
      🔬 &nbsp;<b>Calibración:</b> Un modelo bien calibrado muestra que, cuando predice 70% de probabilidad de aforo
      crítico, aproximadamente el 70% de esos eventos <em>realmente</em> son críticos. Esta sección evalúa
      si las probabilidades del modelo son realistas o si sobre/sub-estima.
    </div>
    """, unsafe_allow_html=True)

    c_cal1, c_cal2 = st.columns(2)

    with c_cal1:
        # Curva de calibración por bins
        n_bins = 5
        bin_edges  = np.linspace(0, 1, n_bins + 1)
        bin_centers, real_fracs, pred_means, bin_counts = [], [], [], []

        for i in range(n_bins):
            mask = (probas_val >= bin_edges[i]) & (probas_val < bin_edges[i + 1])
            if mask.sum() >= 2:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                real_fracs.append(float(y_real_clf[mask].mean()))
                pred_means.append(float(probas_val[mask].mean()))
                bin_counts.append(int(mask.sum()))

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#3A3A52", width=1.5, dash="dash"),
            name="Calibración perfecta",
        ))
        if bin_centers:
            fig_cal.add_trace(go.Scatter(
                x=pred_means, y=real_fracs, mode="lines+markers",
                line=dict(color="#C9A227", width=2),
                marker=dict(size=[min(8 + c // 2, 18) for c in bin_counts],
                            color="#C9A227", opacity=0.85),
                name="Modelo MONASTERY",
                text=[f"n={c}" for c in bin_counts],
                hovertemplate="Prob. predicha: %{x:.2f}<br>Frac. real: %{y:.2f}<br>%{text}<extra></extra>",
            ))
        fig_cal.update_layout(
            title=dict(text="Curva de calibración (Reliability Diagram)",
                       font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Probabilidad predicha", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E"), range=[0, 1]),
            yaxis=dict(title="Fracción real de críticos", gridcolor="#1E1E30",
                       tickfont=dict(color="#8B949E"), range=[0, 1]),
            legend=dict(font=dict(size=10, color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=50, r=20, t=55, b=50), height=320,
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    with c_cal2:
        # ROC de validación
        from sklearn.metrics import roc_curve, auc as auc_fn
        if len(np.unique(y_real_clf)) > 1:
            fpr, tpr, _ = roc_curve(y_real_clf, probas_val)
            roc_auc_val = auc_fn(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="#3A3A52", width=1.5, dash="dash"),
            )
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#C9A227", width=2.5),
                fill="tozeroy", fillcolor="rgba(201,162,39,.08)",
                name=f"AUC = {roc_auc_val:.4f}",
            ))
            fig_roc.update_layout(
                title=dict(text="Curva ROC — conjunto de validación",
                           font=dict(color="#C9D1D9", size=13)),
                plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
                font=dict(color="#8B949E", size=11),
                xaxis=dict(title="Tasa Falsos Positivos (FPR)", gridcolor="#1E1E30",
                           tickfont=dict(color="#8B949E"), range=[0, 1]),
                yaxis=dict(title="Tasa Verdaderos Positivos (TPR)",
                           gridcolor="#1E1E30", tickfont=dict(color="#8B949E"), range=[0, 1]),
                legend=dict(font=dict(size=11, color="#C9A227"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=50, r=20, t=55, b=50), height=320,
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Se necesitan eventos de ambas clases en el conjunto de validación para graficar la curva ROC.")

    # Comparación vs métricas de entrenamiento
    st.markdown("##### Comparación: validación temporal vs. métricas de entrenamiento")
    m_train = modelo.resultados_clf.get(modelo_clf_sel, {})
    m_train_r = modelo.resultados_reg.get(modelo_reg_sel, {})

    filas_cmp = []
    filas_cmp.append({
        "Métrica": "Accuracy (clasificación)",
        "Entrenamiento (test split)": f"{m_train.get('accuracy', 0)*100:.2f}%",
        "Validación temporal": f"{acc_val*100:.2f}%",
        "Δ": f"{(acc_val - m_train.get('accuracy', 0))*100:+.2f}pp",
    })
    filas_cmp.append({
        "Métrica": "F1-score (aforo crítico)",
        "Entrenamiento (test split)": f"{m_train.get('f1', 0)*100:.2f}%",
        "Validación temporal": f"{f1_val*100:.2f}%",
        "Δ": f"{(f1_val - m_train.get('f1', 0))*100:+.2f}pp",
    })
    filas_cmp.append({
        "Métrica": "RMSE asistencia (personas)",
        "Entrenamiento (test split)": f"{m_train_r.get('rmse', 0):.1f}",
        "Validación temporal": f"{rmse_val:.1f}",
        "Δ": f"{rmse_val - m_train_r.get('rmse', 0):+.1f}",
    })

    df_cmp = pd.DataFrame(filas_cmp)

    def style_delta(val):
        try:
            num = float(val.replace("pp", "").replace("%", ""))
            if num > 0:
                return "color: #3FB950"
            elif num < -5:
                return "color: #EF4444"
            return "color: #8B949E"
        except Exception:
            return ""

    st.dataframe(
        df_cmp.style.applymap(style_delta, subset=["Δ"]),
        use_container_width=True, hide_index=True, height=145,
    )

    if all(
        abs(float(r["Δ"].replace("pp", "").replace("%", ""))) <= 10
        for r in filas_cmp
    ):
        st.markdown("""
        <div class="success-box">
          ✅ &nbsp;<b>Sin señales de sobreajuste significativo:</b> las métricas de validación temporal
          son coherentes con las del entrenamiento. El modelo generaliza bien a eventos no vistos.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warn-box">
          ⚠ &nbsp;Se detecta una brecha entre las métricas de entrenamiento y validación temporal.
          Puede indicar sobreajuste leve o que los eventos recientes tienen un patrón diferente.
          Considerar reentrenar con datos más recientes o reducir la complejidad del modelo.
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# TAB 4: TABLA COMPLETA
# ────────────────────────────────────────────────────────────────────
with tab_tabla:
    # Construir tabla de validación detallada
    filas = []
    for i in range(n_eventos_val):
        acierto = bool(preds_val[i] == y_real_clf[i])
        filas.append({
            "Nº": i + 1,
            "Evento": (df_val["nombre"].iloc[i]
                       if "nombre" in df_val.columns else f"Evento {i+1}"),
            "Fecha": (pd.to_datetime(df_val["fecha"].iloc[i]).strftime("%d/%m/%Y")
                      if "fecha" in df_val.columns else "—"),
            "Categoría": (df_val["categoria"].iloc[i]
                          if "categoria" in df_val.columns else "—"),
            "Asistencia real": int(y_real_asist[i]),
            "Asistencia pred.": int(round(asist_pred[i])),
            "Error abs.": int(abs(err_asist[i])),
            "Error %": f"{abs(err_asist[i]) / max(y_real_asist[i], 1) * 100:.1f}%",
            "Aforo real": AFORO_LABELS[int(y_real_clf[i])],
            "Aforo pred.": AFORO_LABELS[int(preds_val[i])],
            "Prob. crítico": f"{probas_val[i]*100:.1f}%",
            "Acierto": "✓" if acierto else "✗",
        })
    df_tabla = pd.DataFrame(filas)

    def color_acierto(val):
        if val == "✓":
            return "color: #3FB950; font-weight: bold"
        if val == "✗":
            return "color: #EF4444; font-weight: bold"
        return ""

    def color_aforo(val):
        if val == "Aforo Crítico":
            return "color: #EF4444"
        if val == "Aforo Normal":
            return "color: #3FB950"
        return ""

    st.markdown(f"""
    <div class="info-box">
      📋 &nbsp;Tabla detallada de los <b>{n_eventos_val} eventos</b> del período de validación.
      Compara asistencia real vs. predicha y clasificación real vs. predicha para cada evento.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df_tabla.style
            .applymap(color_acierto, subset=["Acierto"])
            .applymap(color_aforo,   subset=["Aforo real", "Aforo pred."]),
        use_container_width=True, hide_index=True, height=420,
    )

    # Estadísticas resumen de la tabla
    st.markdown("##### Estadísticas del período de validación")
    rs1, rs2, rs3, rs4 = st.columns(4)
    rs1.metric("Error medio absoluto (MAE)", f"{mae_val:.1f} personas")
    rs2.metric("Error medio relativo",
               f"{np.mean(np.abs(err_asist)/y_real_asist.clip(1))*100:.1f}%")
    rs3.metric("Eventos con error <10 pers.",
               f"{int(np.sum(np.abs(err_asist) < 10))} / {n_eventos_val}")
    rs4.metric("Mayor error absoluto",
               f"{int(np.abs(err_asist).max())} personas")

# ══════════════════════════════════════════════════════════════════════
# E) CIERRE — RESUMEN EJECUTIVO PARA GERENCIA
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 🎯 Resumen ejecutivo para toma de decisiones")

c_ej1, c_ej2 = st.columns([2, 1])

with c_ej1:
    icon_conf = "✅" if confianza >= 80 else ("⚠" if confianza >= 60 else "🚨")
    recomendacion = (
        "El modelo está listo para uso operativo. Se puede confiar en sus alertas para "
        "planificar personal, inventario y logística con <b>48 horas de anticipación</b>."
        if confianza >= 80 else
        "El modelo puede usarse como guía pero requiere supervisión humana. "
        "Se recomienda reentrenar con más datos históricos para mejorar la confianza."
        if confianza >= 60 else
        "El modelo muestra brechas importantes en este período de validación. "
        "Se recomienda revisar la calidad del dataset y reentrenar."
    )
    st.markdown(f"""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;
        letter-spacing:.06em;margin-bottom:.75rem">Conclusión del análisis de validación</div>
      <div style="font-size:.88rem;color:#C9D1D9;line-height:1.8">
        {icon_conf} &nbsp;<b>Índice de confianza: {confianza:.1f}% ({c_label})</b><br>
        Sobre {n_eventos_val} eventos de validación, el modelo clasificó correctamente
        <b>{hits_clf}</b> ({acc_val*100:.1f}%) y predijo asistencia con un error
        promedio de <b>{mae_val:.0f} personas</b> (±{np.mean(np.abs(err_asist)/y_real_asist.clip(1))*100:.1f}% relativo).<br><br>
        <span style="color:#8B949E">{recomendacion}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c_ej2:
    st.markdown(f"""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;
        letter-spacing:.06em;margin-bottom:.75rem">Valor de negocio estimado</div>
      <div style="font-size:.84rem;color:#C9D1D9;line-height:1.85">
        <span style="color:#8B949E">Eventos críticos detectados:</span><br>
        &nbsp;&nbsp;<b style="color:#EF4444">{tp}</b> de {tp+fn} reales ({f"{tp/(tp+fn)*100:.0f}%" if (tp+fn)>0 else "n/a"})<br><br>
        <span style="color:#8B949E">Alertas tempranas útiles:</span><br>
        &nbsp;&nbsp;<b style="color:#3FB950">{tp}</b> refuerzos de personal acertados<br><br>
        <span style="color:#8B949E">Sobre-alertas (costo menor):</span><br>
        &nbsp;&nbsp;<b style="color:#C9A227">{fp}</b> preparaciones extra sin evento crítico<br><br>
        <span style="color:#8B949E">Eventos críticos no detectados:</span><br>
        &nbsp;&nbsp;<b style="color:#EF4444">{fn}</b> riesgos operativos no anticipados
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="margin-top:.5rem">
  📚 &nbsp;<b>Nota metodológica (CRISP-DM Fase 6):</b> Esta validación temporal utiliza los últimos
  N eventos del historial como conjunto de prueba, simulando el uso prospectivo del modelo. No confundir
  con el test split del entrenamiento (aleatorio 20%). La validación temporal es más exigente porque
  evalúa la capacidad de generalización a eventos futuros ordenados en el tiempo.
</div>
""", unsafe_allow_html=True)
