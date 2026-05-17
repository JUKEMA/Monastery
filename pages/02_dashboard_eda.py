"""
pages/02_dashboard_eda.py · Dashboard EDA — MONASTERY Analytics
CRISP-DM Fase 2: Comprensión de los datos
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Dashboard EDA · MONASTERY", layout="wide", page_icon="📊")
st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D0D14}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
[data-testid="metric-container"]{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem!important}
[data-testid="metric-container"] label{color:#8B949E!important;font-size:.74rem!important;font-weight:500;text-transform:uppercase}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:600;color:#E6E0C8!important}
.risk-card{background:#12121C;border:1px solid #1E1E30;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #1E1E30;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6E0C8;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.info-box{background:rgba(201,162,39,.08);border:1px solid rgba(201,162,39,.3);border-radius:8px;padding:.75rem 1rem;color:#E8C84A;font-size:.85rem;margin:.5rem 0}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
</style>""", unsafe_allow_html=True)

if "df_master" not in st.session_state or st.session_state.df_master is None:
    st.warning("⚠ Carga el dataset primero en ⚡ Entrenamiento.")
    st.stop()

df_full = st.session_state.df_master.copy()

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">📊</span>
  <div>
    <h1>Dashboard Exploratorio — EDA</h1>
    <p class="subtitle">CRISP-DM Fase 2 · Análisis de 150 eventos · Asistencia · Ingresos · Clima · Categorías</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Filtros
with st.expander("🔍 Filtros del dashboard", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    cats = ["Todas"] + sorted(df_full["categoria"].dropna().unique().tolist())
    with fc1:
        cat_sel = st.selectbox("Categoría", cats)
    climas = ["Todos"] + sorted(df_full["clima"].dropna().unique().tolist())
    with fc2:
        clima_sel = st.selectbox("Clima", climas)
    with fc3:
        festivo_sel = st.selectbox("Día festivo", ["Todos", "Sí", "No"])

df = df_full.copy()
if cat_sel != "Todas":
    df = df[df["categoria"] == cat_sel]
if clima_sel != "Todos":
    df = df[df["clima"] == clima_sel]
if festivo_sel == "Sí":
    df = df[df["es_festivo"] == 1]
elif festivo_sel == "No":
    df = df[df["es_festivo"] == 0]

if df.empty:
    st.warning("No hay eventos con los filtros seleccionados.")
    st.stop()

# ── KPIs
st.markdown("#### KPIs Generales")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Eventos", len(df))
k2.metric("Aforo Crítico", f"{df['aforo_critico'].sum()} ({df['aforo_critico'].mean()*100:.0f}%)")
k3.metric("Asistencia promedio", f"{df['asistencia_total'].mean():.0f}")
k4.metric("Ocupación promedio", f"{df['tasa_ocupacion'].mean():.1f}%")
k5.metric("Ingreso promedio", f"${df['ingreso_total'].mean()/1e6:.1f}M")
k6.metric("Personal promedio", f"{df['n_personal'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Fila 1: Asistencia por categoría + Ocupación por clima
col1, col2 = st.columns(2)

DIAS = {0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"}
PAL = ["#C9A227","#3FB950","#58A6FF","#8B5CF6","#EF4444","#F97316","#06B6D4",
       "#EC4899","#84CC16","#A78BFA"]

with col1:
    agg = df.groupby("categoria")["asistencia_total"].agg(["mean","std","count"]).reset_index()
    agg.columns = ["categoria","mean","std","count"]
    agg = agg.sort_values("mean", ascending=True)
    fig1 = go.Figure(go.Bar(
        x=agg["mean"], y=agg["categoria"], orientation="h",
        marker_color=PAL[:len(agg)],
        text=[f"{v:.0f}" for v in agg["mean"]], textposition="outside",
        textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig1.update_layout(
        title=dict(text="Asistencia promedio por categoría", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        xaxis=dict(gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
        yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
        margin=dict(l=10, r=60, t=50, b=30), height=340,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    agg2 = df.groupby("clima")["tasa_ocupacion"].mean().reset_index().sort_values("tasa_ocupacion", ascending=False)
    CLIMA_COLOR = {"Soleado":"#F59E0B","Despejado":"#3FB950","Parcial":"#58A6FF",
                   "Nublado":"#8B949E","Ventoso":"#8B5CF6","Lluvioso":"#3B82F6","Tormentoso":"#EF4444"}
    fig2 = go.Figure(go.Bar(
        x=agg2["clima"], y=agg2["tasa_ocupacion"],
        marker_color=[CLIMA_COLOR.get(c,"#888") for c in agg2["clima"]], opacity=0.85,
        text=[f"{v:.1f}%" for v in agg2["tasa_ocupacion"]], textposition="outside",
        textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig2.update_layout(
        title=dict(text="Tasa de ocupación promedio por clima", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", range=[0, 115], tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        margin=dict(l=40, r=20, t=50, b=40), height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Fila 2: Ingresos por mes + Asistencia por día de semana
col3, col4 = st.columns(2)

with col3:
    MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
             7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    agg3 = df.groupby("mes").agg(
        ingreso=("ingreso_total","mean"),
        n=("id_evento","count")).reset_index()
    agg3["mes_label"] = agg3["mes"].map(MESES)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=agg3["mes_label"], y=agg3["ingreso"]/1e6,
        marker_color="#C9A227", opacity=0.8, name="Ingreso prom.",
        text=[f"${v:.1f}M" for v in agg3["ingreso"]/1e6], textposition="outside",
        textfont=dict(size=9, color="#C9D1D9"),
    ))
    fig3.add_trace(go.Scatter(
        x=agg3["mes_label"], y=agg3["n"], mode="lines+markers",
        name="N° eventos", yaxis="y2", line=dict(color="#3FB950", width=2),
        marker=dict(size=6, color="#3FB950"),
    ))
    fig3.update_layout(
        title=dict(text="Ingreso promedio y eventos por mes", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(title="Ingreso (M$)", gridcolor="#1E1E30", tickfont=dict(color="#C9A227")),
        yaxis2=dict(title="N° eventos", overlaying="y", side="right",
                    tickfont=dict(color="#3FB950"), showgrid=False),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=50, r=50, t=50, b=40), height=340,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    agg4 = df.groupby("dia_semana")["asistencia_total"].mean().reset_index()
    agg4["dia_label"] = agg4["dia_semana"].map(DIAS)
    COLORES_DIA = ["#8B949E","#8B949E","#8B949E","#8B949E","#58A6FF","#C9A227","#EF4444"]
    fig4 = go.Figure(go.Bar(
        x=agg4["dia_label"], y=agg4["asistencia_total"],
        marker_color=COLORES_DIA, opacity=0.85,
        text=[f"{v:.0f}" for v in agg4["asistencia_total"]], textposition="outside",
        textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig4.update_layout(
        title=dict(text="Asistencia promedio por día de la semana", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        margin=dict(l=40, r=20, t=50, b=40), height=340,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Fila 3: Dispersión asistencia vs ingreso + Distribución ocupación
col5, col6 = st.columns(2)

with col5:
    fig5 = px.scatter(
        df, x="asistencia_total", y="ingreso_total",
        color="categoria", size="n_personal",
        hover_data=["nombre", "clima", "tasa_ocupacion"],
        color_discrete_sequence=PAL,
    )
    fig5.update_layout(
        title=dict(text="Asistencia vs Ingreso (tamaño = personal)", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        xaxis=dict(gridcolor="#1E1E30", title="Asistencia total", tickfont=dict(color="#8B949E")),
        yaxis=dict(gridcolor="#1E1E30", title="Ingreso total ($)", tickfont=dict(color="#8B949E")),
        legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=9)),
        margin=dict(l=50, r=20, t=50, b=50), height=360,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(
        x=df["tasa_ocupacion"], nbinsx=20,
        marker_color="#C9A227", opacity=0.8, name="Todos",
    ))
    fig6.add_vline(x=85, line_color="#EF4444", line_width=2, line_dash="dash",
                   annotation_text="Umbral crítico (85%)",
                   annotation_font_color="#EF4444", annotation_font_size=11)
    fig6.update_layout(
        title=dict(text="Distribución de tasa de ocupación", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        xaxis=dict(gridcolor="#1E1E30", title="Tasa de ocupación (%)", tickfont=dict(color="#8B949E")),
        yaxis=dict(gridcolor="#1E1E30", title="N° eventos", tickfont=dict(color="#8B949E")),
        margin=dict(l=50, r=20, t=50, b=50), height=360,
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Correlaciones
st.markdown("---")
st.markdown("#### Mapa de correlaciones con asistencia_total")
num_cols = ["asistencia_total","ingreso_total","n_personal","presupuesto",
            "descuento","entradas_preventa","entradas_taquilla",
            "consumo_bebidas","tasa_ocupacion","es_festivo","es_dia_pago",
            "dia_semana","mes"]
num_cols = [c for c in num_cols if c in df.columns]
corr_m = df[num_cols].corr().round(2)
fig_hm = go.Figure(go.Heatmap(
    z=corr_m.values, x=corr_m.columns, y=corr_m.index,
    colorscale=[[0,"#1A0A2E"],[0.5,"#2A2A3E"],[1,"#C9A227"]],
    zmin=-1, zmax=1, text=corr_m.values,
    texttemplate="%{text:.2f}", textfont=dict(size=9),
    colorbar=dict(
        title=dict(text="Correlación", font=dict(color="#8B949E")),
        tickfont=dict(color="#8B949E"),
    ),
    xgap=2, ygap=2,
))
fig_hm.update_layout(
    plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=10),
    xaxis=dict(tickfont=dict(color="#C9D1D9", size=9), side="bottom"),
    yaxis=dict(tickfont=dict(color="#C9D1D9", size=9), autorange="reversed"),
    margin=dict(l=140, r=20, t=20, b=120), height=420,
)
st.plotly_chart(fig_hm, use_container_width=True)
