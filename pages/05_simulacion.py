"""
pages/05_simulacion.py · Simulación de Escalamiento
MONASTERY Analytics · Analítica Prescriptiva
¿Qué pasa si duplicamos la capacidad? Cuellos de botella proyectados.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Simulación · MONASTERY", layout="wide", page_icon="🔮")
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
.warn-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:8px;padding:.75rem 1rem;color:#FCA5A5;font-size:.85rem;margin:.5rem 0}
.cuello{background:#1A0A08;border:1px solid #4D2000;border-radius:8px;padding:.85rem 1rem;margin:.4rem 0}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
</style>""", unsafe_allow_html=True)

if "modelo" not in st.session_state or not st.session_state.modelo.trained:
    st.warning("⚠ Entrena el modelo primero en ⚡ Entrenamiento.")
    st.stop()

modelo = st.session_state.modelo
df_orig = st.session_state.df_master.copy()

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🔮</span>
  <div>
    <h1>Simulación de Escalamiento de Capacidad</h1>
    <p class="subtitle">¿Qué pasa si duplicamos el aforo? · Cuellos de botella proyectados · Impacto en ingresos · Escenarios comparativos</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  🔮 &nbsp;<b>¿Cómo funciona la simulación?</b> El modelo proyecta el comportamiento del club si se aumentara
  la capacidad física (factor de escala). Al escalar, también se escala proporcionalmente la preventa disponible
  y el personal requerido. El modelo predice cuántos eventos pasarían a aforo crítico bajo el nuevo escenario
  y estima el ingreso proyectado con una tasa de ocupación conservadora del 85%.
</div>
""", unsafe_allow_html=True)

# ── Controles de simulación
st.markdown("#### Configura el escenario de escalamiento")
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    factor = st.slider(
        "Factor de escala de capacidad",
        min_value=1.1, max_value=4.0, value=2.0, step=0.1,
        help="1.0 = sin cambio · 2.0 = duplicar aforo · 3.0 = triplicar",
    )
    st.markdown(f"""
    <div class="risk-card" style="margin-top:.5rem">
      <div style="font-size:.68rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem">Escenario seleccionado</div>
      <div style="font-size:2rem;font-weight:700;color:#C9A227;text-align:center">{factor:.1f}x</div>
      <div style="font-size:.8rem;color:#8B949E;text-align:center">
        {"Leve expansión" if factor < 1.5 else "Duplicación" if factor < 2.5 else "Expansión masiva"}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    tasa_ocupacion_sim = st.slider(
        "Tasa de ocupación asumida (%)",
        min_value=50, max_value=100, value=80, step=5,
        help="Qué porcentaje del nuevo aforo se espera llenar en promedio",
    )
    costo_por_asist = st.number_input(
        "Costo operativo por asistente adicional ($)",
        min_value=0, max_value=50000, value=8000, step=1000,
    )

with col_c3:
    modelo_clf = st.selectbox(
        "Modelo de clasificación",
        list(modelo.modelo_clf.keys()),
        index=list(modelo.modelo_clf.keys()).index("Random Forest Clf.")
        if "Random Forest Clf." in modelo.modelo_clf else 0,
    )
    st.markdown(f"""
    <div class="risk-card" style="margin-top:.5rem">
      <div style="font-size:.68rem;color:#8B949E;text-transform:uppercase;font-weight:600;letter-spacing:.05em;margin-bottom:.4rem">Parámetros del modelo</div>
      <div style="font-size:.8rem;color:#C9D1D9;line-height:1.8">
        Umbral crítico: <b>85%</b><br>
        Eventos base: <b>{len(df_orig)}</b><br>
        Ingreso base total: <b>${df_orig['ingreso_total'].sum()/1e9:.2f}B</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Ejecutar simulación", type="primary"):
    with st.spinner("Simulando el nuevo escenario..."):
        df_sim, cuellos = modelo.simular_escalamiento(
            factor_aforo=factor,
            modelo_clf_nombre=modelo_clf,
        )

    if df_sim is None:
        st.error("Error al ejecutar la simulación.")
        st.stop()

    st.session_state["sim_result"] = (df_sim, cuellos, factor, tasa_ocupacion_sim, costo_por_asist)

if "sim_result" in st.session_state:
    df_sim, cuellos, factor_usado, tasa_ocup_usada, costo_asist = st.session_state["sim_result"]

    ingreso_proyectado = df_orig["ingreso_total"].sum() * factor_usado * (tasa_ocup_usada / 100)
    ingreso_actual     = df_orig["ingreso_total"].sum()
    delta_ingreso      = ingreso_proyectado - ingreso_actual
    asistencia_proy    = df_orig["asistencia_total"].mean() * factor_usado * (tasa_ocup_usada / 100)
    costo_adicional    = (asistencia_proy - df_orig["asistencia_total"].mean()) * costo_asist * len(df_orig)
    roi                = (delta_ingreso - costo_adicional) / max(costo_adicional, 1) * 100

    # ── KPIs del escenario
    st.markdown("---")
    st.markdown(f"#### Resultados — Escenario {factor_usado:.1f}x con {tasa_ocup_usada}% ocupación")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Ingresos proyectados", f"${ingreso_proyectado/1e9:.2f}B",
              delta=f"+${delta_ingreso/1e9:.2f}B")
    k2.metric("Eventos críticos ahora", f"{cuellos['pct_eventos_criticos_orig']:.0f}%")
    k3.metric("Eventos críticos simulado", f"{cuellos['pct_eventos_criticos_nuevo']:.0f}%",
              delta=f"{cuellos['delta_criticos']:+.1f}pp")
    k4.metric("Personal crítico estimado", f"{int(cuellos['personal_critico'])} pers/evento")
    k5.metric("Costo adicional estimado", f"${costo_adicional/1e9:.2f}B")
    k6.metric("ROI estimado", f"{roi:.0f}%",
              delta="positivo" if roi > 0 else "negativo",
              delta_color="normal" if roi > 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparativa original vs simulado
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Distribución de probabilidad de aforo crítico: original vs simulado
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_sim["prob_critico_orig"].fillna(50),
            name="Escenario actual", nbinsx=15,
            marker_color="#3FB950", opacity=0.7,
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_sim["prob_critico_sim"].fillna(50),
            name=f"Escenario {factor_usado:.1f}x", nbinsx=15,
            marker_color="#C9A227", opacity=0.7,
        ))
        fig_dist.add_vline(x=85, line_color="#EF4444", line_dash="dash", line_width=2,
                           annotation_text="Umbral crítico",
                           annotation_font_color="#EF4444", annotation_font_size=10)
        fig_dist.update_layout(
            barmode="overlay",
            title=dict(text="Distribución de probabilidad de aforo crítico", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Probabilidad aforo crítico (%)", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
            yaxis=dict(title="N° eventos", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
            legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
            margin=dict(l=50, r=20, t=50, b=50), height=360,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_s2:
        # Escenarios de ingreso: múltiples factores
        factores = np.arange(1.0, 4.1, 0.25)
        ingresos = [ingreso_actual * f * (tasa_ocup_usada / 100) for f in factores]
        costos   = [
            (df_orig["asistencia_total"].mean() * f * (tasa_ocup_usada/100)
             - df_orig["asistencia_total"].mean()) * costo_asist * len(df_orig)
            for f in factores
        ]
        utilidades = [i - c for i, c in zip(ingresos, costos)]

        fig_esc = go.Figure()
        fig_esc.add_trace(go.Scatter(
            x=factores, y=[i/1e9 for i in ingresos],
            mode="lines+markers", name="Ingreso proyectado",
            line=dict(color="#C9A227", width=2.5), marker=dict(size=6),
        ))
        fig_esc.add_trace(go.Scatter(
            x=factores, y=[c/1e9 for c in costos],
            mode="lines+markers", name="Costo adicional",
            line=dict(color="#EF4444", width=2, dash="dot"), marker=dict(size=6),
        ))
        fig_esc.add_trace(go.Scatter(
            x=factores, y=[u/1e9 for u in utilidades],
            mode="lines+markers", name="Utilidad neta",
            line=dict(color="#3FB950", width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
        ))
        fig_esc.add_vline(x=factor_usado, line_color="#C9A227", line_dash="dash",
                          annotation_text=f"Escenario actual ({factor_usado:.1f}x)",
                          annotation_font_color="#C9A227", annotation_font_size=10)
        fig_esc.update_layout(
            title=dict(text="Curva de ingreso vs costo por factor de escala", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
            xaxis=dict(title="Factor de escala", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
            yaxis=dict(title="Valor (B$)", gridcolor="#1E1E30", tickfont=dict(color="#8B949E")),
            legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
            margin=dict(l=50, r=20, t=50, b=50), height=360,
        )
        st.plotly_chart(fig_esc, use_container_width=True)

    # ── Cuellos de botella
    st.markdown("---")
    st.markdown("#### Cuellos de botella proyectados")
    st.markdown(f"""
    <div class="warn-box">
      ⚠️ Al escalar {factor_usado:.1f}x la capacidad, el modelo proyecta que los eventos críticos
      pasarían del <b>{cuellos['pct_eventos_criticos_orig']:.0f}%</b> al
      <b>{cuellos['pct_eventos_criticos_nuevo']:.0f}%</b> del total (+{cuellos['delta_criticos']:.1f}pp).
      Esto implica mayor demanda de personal de seguridad, logística y barra en simultáneo.
    </div>
    """, unsafe_allow_html=True)

    CUELLOS_INFO = [
        ("🚨 Seguridad y control de acceso",
         f"Con {cuellos['personal_critico']} personas promedio en eventos críticos y un factor {factor_usado:.1f}x, "
         f"se necesitarán al menos {int(cuellos['personal_critico'] * factor_usado)} personas de seguridad. "
         "Este es el primer cuello de botella: contratar temporales con suficiente anticipación.",
         "#EF4444"),
        ("🍹 Inventario de bebidas y barra",
         f"Un aumento de {factor_usado:.1f}x en asistencia implica {factor_usado:.1f}x en consumo de bebidas. "
         "El inventario debe reabastecerse antes del evento. La cadena de frío y el almacenamiento "
         "serán el segundo cuello de botella si no se planea con 72 horas de anticipación.",
         "#F97316"),
        ("🚗 Accesos y parqueadero",
         f"El flujo de personas se incrementa {factor_usado:.1f}x. Los accesos actuales pueden saturarse "
         "en los primeros 30 minutos. Se recomienda gestionar accesos escalonados y coordinar con "
         "parqueaderos aledaños antes de superar 1.5x la capacidad actual.",
         "#C9A227"),
        ("📱 Atención al cliente y taquilla",
         "La taquilla y los sistemas de pago (datafono, efectivo) colapsan antes que la producción. "
         f"Con {factor_usado:.1f}x de demanda se requieren al menos {max(2, int(factor_usado * 2))} puntos de pago activos "
         "y un sistema de pre-registro digital para reducir filas.",
         "#58A6FF"),
        ("🔊 Infraestructura técnica (sonido y luces)",
         "La infraestructura técnica (equipos de sonido, iluminación, camerinos) fue diseñada para el aforo original. "
         f"A partir de {min(factor_usado, 2.0):.1f}x se recomienda auditoría técnica y contratación de equipos adicionales.",
         "#8B5CF6"),
    ]

    for titulo, desc, color in CUELLOS_INFO:
        st.markdown(f"""
        <div class="cuello">
          <div style="font-size:.82rem;color:{color};font-weight:600;margin-bottom:.3rem">{titulo}</div>
          <div style="font-size:.82rem;color:#C9D1D9;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Comparativa por categoría
    st.markdown("---")
    st.markdown("#### Impacto por categoría de evento")

    agg_cat_sim = df_sim.groupby("categoria").agg(
        prob_orig =("prob_critico_orig","mean"),
        prob_sim  =("prob_critico_sim","mean"),
        delta     =("delta_prob","mean"),
        n         =("id_evento","count"),
    ).reset_index().sort_values("delta", ascending=False)

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        name="Prob. actual (%)",
        x=agg_cat_sim["categoria"], y=agg_cat_sim["prob_orig"],
        marker_color="#3FB950", opacity=0.85,
    ))
    fig_cat.add_trace(go.Bar(
        name=f"Prob. escenario {factor_usado:.1f}x (%)",
        x=agg_cat_sim["categoria"], y=agg_cat_sim["prob_sim"],
        marker_color="#C9A227", opacity=0.85,
    ))
    fig_cat.add_hline(y=85, line_color="#EF4444", line_dash="dash",
                      annotation_text="Umbral crítico (85%)",
                      annotation_font_color="#EF4444")
    fig_cat.update_layout(
        barmode="group",
        title=dict(text="Probabilidad de aforo crítico: actual vs simulado por categoría", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#1E1E30", range=[0,115], title="Probabilidad (%)", tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9", size=9), tickangle=-20),
        legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=60, r=20, t=60, b=70), height=380,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # ── Tabla exportable
    st.markdown("##### Detalle por evento")
    df_exp = df_sim[["nombre","categoria","prob_critico_orig","prob_critico_sim","delta_prob"]].copy()
    df_exp.columns = ["Evento","Categoría","Prob. actual (%)","Prob. simulada (%)","Δ Prob. (pp)"]
    df_exp = df_exp.sort_values("Δ Prob. (pp)", ascending=False)
    csv_sim = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar resultados simulación CSV", csv_sim,
                       f"simulacion_{factor_usado:.1f}x_monastery.csv", "text/csv")
    st.dataframe(
        df_exp.style.format({
            "Prob. actual (%)": "{:.1f}","Prob. simulada (%)": "{:.1f}","Δ Prob. (pp)": "{:+.1f}"}),
        use_container_width=True, hide_index=True, height=380,
    )
