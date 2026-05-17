"""
pages/01_entrenamiento.py · Entrenamiento — MONASTERY Analytics
CRISP-DM Fases 3 y 4: Preparación de datos y Modelado
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_monastery import ModeloMonastery, FEATURE_DISPLAY

st.set_page_config(page_title="Entrenamiento · MONASTERY", layout="wide", page_icon="⚡")

CSS = """
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D0D14}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
[data-testid="metric-container"]{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem!important}
[data-testid="metric-container"] label{color:#8B949E!important;font-size:.74rem!important;font-weight:500;text-transform:uppercase;letter-spacing:.05em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:600;color:#E6E0C8!important}
.risk-card{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #1E1E30;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6E0C8;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.info-box{background:rgba(201,162,39,.08);border:1px solid rgba(201,162,39,.3);border-radius:8px;padding:.75rem 1rem;color:#E8C84A;font-size:.85rem;margin:.5rem 0}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

if "modelo" not in st.session_state:
    st.session_state.modelo = ModeloMonastery()
if "df_master" not in st.session_state:
    st.session_state.df_master = None

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">⚡</span>
  <div>
    <h1>Entrenamiento de Modelos</h1>
    <p class="subtitle">CRISP-DM Fases 3 y 4 · Merge esquema estrella · Regresión + Clasificación binaria · CV-5</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 1. Carga del Excel
st.markdown("#### 1 · Cargar Dataset")
uploaded = st.file_uploader(
    "Sube el archivo **dataset_eventos_v3.xlsx**",
    type=["xlsx"],
    help="Archivo con 9 hojas: FACT_EVENTOS, FACT_ASISTENCIA, FACT_VENTAS, DIM_*, PIVOTE_*",
)

if uploaded:
    try:
        with st.spinner("Leyendo y unificando las 9 tablas del esquema estrella..."):
            df = ModeloMonastery.cargar_excel(uploaded)
        st.session_state.df_master = df
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Eventos", f"{len(df):,}")
        c2.metric("Variables", len(df.columns))
        c3.metric("Aforo Crítico", f"{df['aforo_critico'].sum()} ({df['aforo_critico'].mean()*100:.0f}%)")
        c4.metric("Tasa Ocup. promedio", f"{df['tasa_ocupacion'].mean():.1f}%")
        c5.metric("Ingreso total", f"${df['ingreso_total'].sum()/1e9:.2f}B")

        with st.expander("Vista previa del dataset unificado (10 filas)"):
            st.dataframe(
                df[["nombre","categoria","clima","dia_semana","es_festivo",
                    "n_personal","asistencia_total","tasa_ocupacion",
                    "ingreso_total","aforo_critico"]].head(10),
                use_container_width=True, hide_index=True,
            )
        st.success("✅ Dataset cargado correctamente — 9 tablas unificadas en 1 dataset maestro.")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

st.markdown("---")

# ── 2. Configuración
st.markdown("#### 2 · Configuración del Entrenamiento")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Variables predictoras</div>
      <div style="font-size:.82rem;color:#C9D1D9;line-height:1.9">
        ✓ Categoría del evento<br>
        ✓ Género musical<br>
        ✓ Condición climática<br>
        ✓ Festivo / Día de pago<br>
        ✓ Día de semana / Mes<br>
        ✓ Presupuesto · Descuento<br>
        ✓ Personal asignado<br>
        ✓ Entradas en preventa
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Variables objetivo</div>
      <div style="font-size:.82rem;color:#C9D1D9;line-height:1.9">
        <span style="color:#C9A227">◆ Regresión</span><br>
        &nbsp;&nbsp;asistencia_total (número)<br><br>
        <span style="color:#EF4444">■ Clasificación</span><br>
        &nbsp;&nbsp;aforo_critico (0/1)<br>
        &nbsp;&nbsp;Umbral: ocupación ≥ 85%
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#58A6FF;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Pipeline ML</div>
      <div style="font-size:.82rem;color:#C9D1D9;line-height:1.9">
        ✓ Imputación por mediana<br>
        ✓ Label Encoding categóricas<br>
        ✓ StandardScaler (LR)<br>
        ✓ Split 80/20 estratificado<br>
        ✓ CV-5 estratificado<br>
        ✓ class_weight balanced (clf)
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── 3. Botón entrenar
st.markdown("#### 3 · Entrenar")
if st.session_state.df_master is None:
    st.warning("⚠ Primero carga el dataset Excel.")
else:
    if st.button("⚡ Entrenar todos los modelos", type="primary"):
        prog_bar = st.progress(0)
        status   = st.empty()

        def cb(pct, msg):
            prog_bar.progress(pct)
            status.markdown(f"<span style='color:#C9A227;font-size:.85rem'>{msg}</span>",
                            unsafe_allow_html=True)

        try:
            res_r, res_c = modelo.entrenar(
                st.session_state.df_master, progress_cb=cb)
            st.session_state.modelo = modelo
            prog_bar.progress(100)
            status.empty()
            st.success("✅ ¡Entrenamiento completado! Todos los modelos listos.")
        except Exception as e:
            st.error(f"Error durante el entrenamiento: {e}")

# ── 4. Resultados
if modelo.trained:
    st.markdown("---")
    st.markdown("#### 4 · Resultados del Entrenamiento")

    tab_r, tab_c = st.tabs(["📈 Modelos de Regresión", "🎯 Modelos de Clasificación"])

    COLORS = {"Regresión Lineal": "#8B5CF6", "Random Forest Reg.": "#C9A227",
               "XGBoost Reg.": "#3FB950", "Gradient Boosting": "#58A6FF",
               "Regresión Logística": "#8B5CF6", "Random Forest Clf.": "#C9A227",
               "XGBoost Clf.": "#3FB950"}

    with tab_r:
        # Tabla de regresión
        rows_r = []
        for nom, m in modelo.resultados_reg.items():
            rows_r.append({"Modelo": nom, "RMSE": m["rmse"], "MAE": m["mae"],
                           "R²": m["r2"], "CV-5 R² media": m["cv_mean"],
                           "CV-5 R² std": m["cv_std"]})
        df_r = pd.DataFrame(rows_r)

        def hl_best_r(col):
            if col.name in ["R²", "CV-5 R² media"]:
                mx = pd.to_numeric(col, errors="coerce").max()
                return ["background-color:rgba(201,162,39,.15);color:#E8C84A"
                        if v == mx else "" for v in pd.to_numeric(col, errors="coerce")]
            if col.name in ["RMSE", "MAE"]:
                mn = pd.to_numeric(col, errors="coerce").min()
                return ["background-color:rgba(63,185,80,.15);color:#6EE7B7"
                        if v == mn else "" for v in pd.to_numeric(col, errors="coerce")]
            return [""] * len(col)

        st.dataframe(df_r.style.apply(hl_best_r).format(
            {"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.4f}",
             "CV-5 R² media": "{:.4f}", "CV-5 R² std": "{:.4f}"}),
            use_container_width=True, hide_index=True, height=180)

        # Predicho vs Real — mejor modelo
        mejor_r = max(modelo.resultados_reg, key=lambda k: modelo.resultados_reg[k]["r2"])
        mr = modelo.resultados_reg[mejor_r]
        col_rv1, col_rv2 = st.columns(2)

        with col_rv1:
            fig_pv = go.Figure()
            y_t = np.array(mr["y_test"])
            y_p = np.array(mr["y_pred"])
            fig_pv.add_trace(go.Scatter(x=y_t, y=y_p, mode="markers",
                marker=dict(color="#C9A227", size=6, opacity=0.7), name="Eventos"))
            lim = max(y_t.max(), y_p.max()) * 1.05
            fig_pv.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                line=dict(color="#3A3A52", width=1, dash="dash"))
            fig_pv.update_layout(
                title=dict(text=f"Real vs Predicho — {mejor_r}", font=dict(color="#C9D1D9", size=13)),
                plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
                font=dict(color="#8B949E", size=11),
                xaxis=dict(title="Asistencia real", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
                yaxis=dict(title="Asistencia predicha", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
                margin=dict(l=50, r=20, t=50, b=50), height=320,
            )
            st.plotly_chart(fig_pv, use_container_width=True)

        with col_rv2:
            # Importancia de variables regresión
            if modelo.importancia_reg:
                labels = [FEATURE_DISPLAY.get(k, k) for k in modelo.importancia_reg]
                vals   = list(modelo.importancia_reg.values())
                fig_ir = go.Figure(go.Bar(
                    x=vals[::-1], y=labels[::-1], orientation="h",
                    marker_color="#C9A227", opacity=0.85,
                    text=[f"{v:.4f}" for v in vals[::-1]],
                    textposition="outside", textfont=dict(size=9, color="#C9D1D9"),
                ))
                fig_ir.update_layout(
                    title=dict(text="Importancia de variables (Regresión)", font=dict(color="#C9D1D9", size=13)),
                    plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
                    font=dict(color="#8B949E", size=11),
                    xaxis=dict(gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
                    yaxis=dict(tickfont=dict(color="#C9D1D9", size=9)),
                    margin=dict(l=10, r=60, t=50, b=30), height=320,
                )
                st.plotly_chart(fig_ir, use_container_width=True)

    with tab_c:
        rows_c = []
        for nom, m in modelo.resultados_clf.items():
            rows_c.append({"Modelo": nom, "Accuracy": m["accuracy"],
                           "Precision": m["precision"], "Recall": m["recall"],
                           "F1": m["f1"], "ROC-AUC": m.get("roc_auc", 0),
                           "CV-5 F1": m["cv_mean"]})
        df_c = pd.DataFrame(rows_c)

        def hl_best_c(col):
            if col.name in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV-5 F1"]:
                try:
                    mx = pd.to_numeric(col, errors="coerce").max()
                    return ["background-color:rgba(201,162,39,.15);color:#E8C84A"
                            if v == mx else "" for v in pd.to_numeric(col, errors="coerce")]
                except Exception:
                    return [""] * len(col)
            return [""] * len(col)

        st.dataframe(df_c.style.apply(hl_best_c).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "CV-5 F1": "{:.4f}"}),
            use_container_width=True, hide_index=True, height=160)

        # Matrices de confusión
        st.markdown("##### Matrices de confusión")
        cols_cm = st.columns(len(modelo.resultados_clf))
        for col_cm, (nombre, m) in zip(cols_cm, modelo.resultados_clf.items()):
            with col_cm:
                cm = np.array(m["cm"])
                cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
                txt  = [[f"{cm[i][j]}<br><span style='font-size:.7rem'>({cm_n[i][j]*100:.0f}%)</span>"
                         for j in range(2)] for i in range(2)]
                fig_cm = go.Figure(go.Heatmap(
                    z=cm_n, x=["Normal (pred)", "Crítico (pred)"],
                    y=["Normal (real)", "Crítico (real)"],
                    text=txt, texttemplate="%{text}",
                    colorscale=[[0,"#0D0D14"],[0.5,"#2A1A00"],[1,"#C9A227"]],
                    showscale=False, xgap=3, ygap=3,
                ))
                fig_cm.update_layout(
                    title=dict(text=nombre, font=dict(color="#C9D1D9", size=11)),
                    plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
                    font=dict(color="#8B949E", size=10),
                    xaxis=dict(tickfont=dict(color="#8B949E", size=9)),
                    yaxis=dict(tickfont=dict(color="#8B949E", size=9), autorange="reversed"),
                    margin=dict(l=80, r=10, t=50, b=60), height=260,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Importancia clasificación
        if modelo.importancia_clf:
            st.markdown("##### Importancia de variables (Clasificación)")
            labels_c = [FEATURE_DISPLAY.get(k, k) for k in modelo.importancia_clf]
            vals_c   = list(modelo.importancia_clf.values())
            pal = ["#C9A227","#3FB950","#58A6FF","#8B5CF6","#EF4444",
                   "#F97316","#06B6D4","#EC4899","#84CC16","#A78BFA","#34D399"]
            fig_ic = go.Figure(go.Bar(
                x=vals_c[::-1], y=labels_c[::-1], orientation="h",
                marker_color=pal[:len(labels_c)][::-1], opacity=0.9,
                text=[f"{v:.4f}" for v in vals_c[::-1]],
                textposition="outside", textfont=dict(size=9, color="#C9D1D9"),
            ))
            fig_ic.update_layout(
                title=dict(text="Importancia de variables (RF Clasificador)", font=dict(color="#C9D1D9", size=13)),
                plot_bgcolor="#0D0D14", paper_bgcolor="#12121C",
                font=dict(color="#8B949E", size=11),
                xaxis=dict(gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
                yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
                margin=dict(l=10, r=70, t=50, b=30), height=340,
            )
            st.plotly_chart(fig_ic, use_container_width=True)
