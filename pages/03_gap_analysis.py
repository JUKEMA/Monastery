"""
pages/03_gap_analysis.py · Análisis de Brechas (Gap Analysis)
MONASTERY Analytics · CRISP-DM Fase 5: Evaluación
Contrasta el desempeño real de cada evento contra el máximo teórico posible.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Gap Analysis · MONASTERY", layout="wide", page_icon="📉")
st.markdown("""
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
.warn-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:8px;padding:.75rem 1rem;color:#FCA5A5;font-size:.85rem;margin:.5rem 0}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
</style>""", unsafe_allow_html=True)

if "df_master" not in st.session_state or st.session_state.df_master is None:
    st.warning("⚠ Carga el dataset primero en ⚡ Entrenamiento.")
    st.stop()

df = st.session_state.df_master.copy()

# ── KPIs derivados para gap analysis
df["aforo_max_teorico"] = df.groupby("categoria")["asistencia_total"].transform("max")
df["brecha_abs"]        = df["aforo_max_teorico"] - df["asistencia_total"]
df["pct_gap"]           = (df["brecha_abs"] / df["aforo_max_teorico"].replace(0, np.nan) * 100).round(1)
df["ingreso_perdido"]   = (df["brecha_abs"] * df["consumo_por_asistente"].fillna(0)).round(0)
df["eficiencia"]        = (100 - df["pct_gap"]).round(1)

PAL = ["#C9A227","#3FB950","#58A6FF","#8B5CF6","#EF4444",
       "#F97316","#06B6D4","#EC4899","#84CC16","#A78BFA"]

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">📉</span>
  <div>
    <h1>Análisis de Brechas de Aforo (Gap Analysis)</h1>
    <p class="subtitle">Aforo real vs. máximo teórico histórico · Ingreso no capturado · Eficiencia por categoría y clima</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  📐 &nbsp;<b>Metodología del Gap Analysis:</b> El "máximo teórico" se define como el mayor
  nivel de asistencia históricamente registrado para cada categoría de evento. La brecha
  representa la diferencia entre ese máximo y la asistencia real de cada evento.
  El ingreso no capturado se estima multiplicando la brecha de asistentes por el consumo
  promedio por persona del evento correspondiente.
</div>
""", unsafe_allow_html=True)

# ── KPIs globales
st.markdown("#### KPIs de Brecha Global")
brecha_total   = int(df["brecha_abs"].sum())
ingreso_perdido= int(df["ingreso_perdido"].sum())
eficiencia_prom= df["eficiencia"].mean()
eventos_100    = (df["pct_gap"] == 0).sum()
mayor_brecha   = df.loc[df["brecha_abs"].idxmax(), "categoria"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Brecha total (asistentes)", f"{brecha_total:,}")
k2.metric("Ingreso no capturado estimado", f"${ingreso_perdido/1e6:.1f}M")
k3.metric("Eficiencia promedio", f"{eficiencia_prom:.1f}%")
k4.metric("Eventos al 100% de capacidad", f"{eventos_100}")
k5.metric("Categoría con mayor brecha", mayor_brecha)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECCIÓN 1: Brecha por categoría
# ─────────────────────────────────────────────────────────
st.markdown("#### 1 · Brecha promedio por categoría de evento")
col1, col2 = st.columns(2)

with col1:
    agg_cat = df.groupby("categoria").agg(
        asistencia_real=("asistencia_total", "mean"),
        maximo_teorico=("aforo_max_teorico", "max"),
        brecha=("brecha_abs", "mean"),
        pct_gap=("pct_gap", "mean"),
        n=("id_evento","count"),
    ).reset_index().sort_values("pct_gap", ascending=False)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Aforo máximo teórico",
        x=agg_cat["categoria"], y=agg_cat["maximo_teorico"],
        marker_color="rgba(201,162,39,0.3)", marker_line_color="#C9A227",
        marker_line_width=1,
    ))
    fig1.add_trace(go.Bar(
        name="Asistencia real promedio",
        x=agg_cat["categoria"], y=agg_cat["asistencia_real"],
        marker_color="#C9A227", opacity=0.9,
        text=[f"{v:.0f}" for v in agg_cat["asistencia_real"]],
        textposition="outside", textfont=dict(size=9, color="#C9D1D9"),
    ))
    fig1.update_layout(
        barmode="overlay",
        title=dict(text="Real vs. Máximo teórico por categoría", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", tickfont=dict(color="#8B949E"), title="Asistentes"),
        xaxis=dict(tickfont=dict(color="#C9D1D9", size=9), tickangle=-30),
        legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=50, r=20, t=50, b=80), height=380,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Bar(
        x=agg_cat["pct_gap"],
        y=agg_cat["categoria"],
        orientation="h",
        marker_color=[PAL[i % len(PAL)] for i in range(len(agg_cat))],
        text=[f"{v:.1f}%" for v in agg_cat["pct_gap"]],
        textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
        customdata=agg_cat[["n", "brecha"]].values,
        hovertemplate="<b>%{y}</b><br>Brecha: %{x:.1f}%<br>Eventos: %{customdata[0]}<br>Brecha media: %{customdata[1]:.0f} asist.<extra></extra>",
    ))
    fig2.update_layout(
        title=dict(text="% de brecha promedio por categoría", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        xaxis=dict(gridcolor="#1E1E30", title="Brecha (%)", tickfont=dict(color="#8B949E")),
        yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
        margin=dict(l=10, r=60, t=50, b=30), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECCIÓN 2: Brecha por clima y día
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 2 · Impacto del clima y el día de la semana en la brecha")
col3, col4 = st.columns(2)

DIAS = {0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"}
CLIMA_COLOR = {"Soleado":"#F59E0B","Despejado":"#3FB950","Parcial":"#58A6FF",
               "Nublado":"#8B949E","Ventoso":"#8B5CF6","Lluvioso":"#3B82F6","Tormentoso":"#EF4444"}

with col3:
    agg_clima = df.groupby("clima").agg(
        pct_gap=("pct_gap","mean"),
        asistencia=("asistencia_total","mean"),
        n=("id_evento","count"),
    ).reset_index().sort_values("pct_gap", ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Brecha (%)",
        x=agg_clima["clima"], y=agg_clima["pct_gap"],
        marker_color=[CLIMA_COLOR.get(c,"#888") for c in agg_clima["clima"]],
        opacity=0.85,
        text=[f"{v:.1f}%" for v in agg_clima["pct_gap"]],
        textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig3.update_layout(
        title=dict(text="Brecha de aforo promedio por clima", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", title="Brecha (%)", tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        margin=dict(l=50, r=20, t=50, b=50), height=320,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    agg_dia = df.groupby("dia_semana").agg(
        pct_gap=("pct_gap","mean"),
        eficiencia=("eficiencia","mean"),
    ).reset_index()
    agg_dia["dia_label"] = agg_dia["dia_semana"].map(DIAS)
    COLORES_DIA = ["#8B949E","#8B949E","#8B949E","#8B949E","#58A6FF","#C9A227","#EF4444"]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=agg_dia["dia_label"], y=agg_dia["pct_gap"],
        marker_color=COLORES_DIA, opacity=0.85, name="Brecha (%)",
        text=[f"{v:.1f}%" for v in agg_dia["pct_gap"]],
        textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig4.add_trace(go.Scatter(
        x=agg_dia["dia_label"], y=agg_dia["eficiencia"],
        mode="lines+markers", name="Eficiencia (%)", yaxis="y2",
        line=dict(color="#3FB950", width=2), marker=dict(size=7, color="#3FB950"),
    ))
    fig4.update_layout(
        title=dict(text="Brecha y eficiencia por día de la semana", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", title="Brecha (%)", tickfont=dict(color="#C9A227")),
        yaxis2=dict(title="Eficiencia (%)", overlaying="y", side="right",
                    tickfont=dict(color="#3FB950"), showgrid=False),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=50, r=60, t=50, b=40), height=320,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECCIÓN 3: Ingreso no capturado
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 3 · Ingreso potencial no capturado por evento")

agg_ing = df.groupby("categoria").agg(
    ingreso_perdido=("ingreso_perdido","sum"),
    n=("id_evento","count"),
    brecha_prom=("brecha_abs","mean"),
).reset_index().sort_values("ingreso_perdido", ascending=False)

col5, col6 = st.columns([2, 1])
with col5:
    fig5 = go.Figure(go.Bar(
        x=agg_ing["categoria"], y=agg_ing["ingreso_perdido"]/1e6,
        marker_color=PAL[:len(agg_ing)], opacity=0.9,
        text=[f"${v:.1f}M" for v in agg_ing["ingreso_perdido"]/1e6],
        textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
        customdata=agg_ing[["n","brecha_prom"]].values,
        hovertemplate="<b>%{x}</b><br>Ingreso no capturado: $%{y:.2f}M<br>N° eventos: %{customdata[0]}<br>Brecha media: %{customdata[1]:.0f} asist.<extra></extra>",
    ))
    fig5.update_layout(
        title=dict(text="Ingreso potencial no capturado por categoría (total 2024)", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", title="Ingreso estimado (M$)", tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9", size=9), tickangle=-20),
        margin=dict(l=60, r=20, t=50, b=70), height=360,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.markdown("""
    <div class="risk-card" style="margin-top:1rem">
      <div style="font-size:.7rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.8rem">
        🏆 Ranking de oportunidad
      </div>
    """, unsafe_allow_html=True)
    for i, row in agg_ing.iterrows():
        color = PAL[list(agg_ing.index).index(i) % len(PAL)]
        st.markdown(f"""
        <div style="padding:.4rem 0;border-bottom:1px solid #1E1E30">
          <div style="font-size:.82rem;color:{color};font-weight:600">{row['categoria']}</div>
          <div style="font-size:.78rem;color:#8B949E">
            ${row['ingreso_perdido']/1e6:.1f}M no capturado · {row['n']} eventos · {row['brecha_prom']:.0f} asist/evento de brecha
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECCIÓN 4: Waterfall — Composición de la brecha
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 4 · Cascada de valor — Descomposición de la brecha total")

cat_ord = agg_cat.sort_values("brecha", ascending=False)
labels  = list(cat_ord["categoria"]) + ["TOTAL"]
values  = list(cat_ord["brecha"].round(0)) + [cat_ord["brecha"].sum().round(0)]
colors_wf = ["#EF4444" if v > 0 else "#3FB950" for v in values[:-1]] + ["#C9A227"]
measure = ["relative"] * (len(labels) - 1) + ["total"]

fig6 = go.Figure(go.Waterfall(
    name="Brecha",
    orientation="v",
    measure=measure,
    x=labels,
    y=values,
    connector=dict(line=dict(color="#2A2A3E", width=1)),
    decreasing=dict(marker_color="#3FB950"),
    increasing=dict(marker_color="#EF4444"),
    totals=dict(marker_color="#C9A227"),
    text=[f"{v:.0f}" for v in values],
    textposition="outside",
    textfont=dict(color="#C9D1D9", size=10),
))
fig6.update_layout(
    title=dict(text="Cascada de brecha de asistencia promedio por categoría (asistentes por evento)", font=dict(color="#C9D1D9", size=13)),
    plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
    yaxis=dict(gridcolor="#1E1E30", title="Asistentes por evento", tickfont=dict(color="#8B949E")),
    xaxis=dict(tickfont=dict(color="#C9D1D9", size=10), tickangle=-15),
    margin=dict(l=60, r=20, t=60, b=60), height=380,
)
st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECCIÓN 5: Tabla detallada de eventos con mayor brecha
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 5 · Top 20 eventos con mayor brecha de aforo")

cols_tabla = ["nombre","categoria","clima","dia_semana","asistencia_total",
              "aforo_max_teorico","brecha_abs","pct_gap","ingreso_perdido","eficiencia"]
df_top = df[cols_tabla].sort_values("brecha_abs", ascending=False).head(20).copy()
df_top["dia_semana"] = df_top["dia_semana"].map(DIAS)
df_top.columns = ["Evento","Categoría","Clima","Día","Asistencia real",
                  "Máx. teórico","Brecha (asist.)","Brecha (%)","Ingreso perdido ($)","Eficiencia (%)"]

def color_brecha(val):
    if isinstance(val, float):
        if val > 60: return "color:#EF4444"
        if val > 30: return "color:#F59E0B"
        return "color:#3FB950"
    return ""

st.dataframe(
    df_top.style.map(color_brecha, subset=["Brecha (%)"]).format({
        "Asistencia real": "{:,.0f}", "Máx. teórico": "{:,.0f}",
        "Brecha (asist.)": "{:,.0f}", "Brecha (%)": "{:.1f}",
        "Ingreso perdido ($)": "${:,.0f}", "Eficiencia (%)": "{:.1f}",
    }),
    use_container_width=True, hide_index=True, height=560,
)

ingreso_recuperable = df.nlargest(20,"brecha_abs")["ingreso_perdido"].sum()
st.markdown(f"""
<div class="warn-box">
  ⚠️ &nbsp;Los <b>20 eventos con mayor brecha</b> acumulan un ingreso potencial no capturado de
  <b>${ingreso_recuperable/1e6:.1f}M</b>. Intervenir con estrategias de marketing anticipado
  en estas fechas podría recuperar hasta un <b>30–50%</b> de esa brecha histórica.
</div>
""", unsafe_allow_html=True)
