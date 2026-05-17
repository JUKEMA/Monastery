"""
pages/04_prescripcion.py · Prescripción de Acciones y Valor del Cliente
MONASTERY Analytics · CRISP-DM Fase 6: Comunicación de resultados
Predice aforo de un evento futuro y prescribe acciones operativas concretas.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_monastery import (
    ModeloMonastery, CLIMA_MAP, CATEGORIA_MAP, GENERO_MAP,
    FEATURE_DISPLAY, AFORO_LABELS, AFORO_COLORS, UMBRAL_CRITICO,
)

st.set_page_config(page_title="Prescripción · MONASTERY", layout="wide", page_icon="🎯")
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
.action-item{padding:.5rem .75rem;margin:.35rem 0;border-radius:6px;border-left:3px solid;font-size:.86rem}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D0D14}::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:3px}
</style>""", unsafe_allow_html=True)

if "modelo" not in st.session_state or not st.session_state.modelo.trained:
    st.warning("⚠ Entrena el modelo primero en ⚡ Entrenamiento.")
    st.stop()

modelo = st.session_state.modelo
df     = st.session_state.df_master

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🎯</span>
  <div>
    <h1>Prescripción de Acciones por Evento</h1>
    <p class="subtitle">Predicción de aforo crítico · Acciones por umbral de probabilidad · Lealtad y valor del cliente · Ranking de eventos</p>
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔮 Predicción individual", "📋 Ranking de eventos", "💎 Lealtad y valor del cliente"])

# ═════════════════════════════════════════════════════════
# TAB 1: Predicción individual
# ═════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("#### Configura el próximo evento para obtener la predicción y prescripción")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("**Características del evento**")
        categoria = st.selectbox("Categoría del evento", list(CATEGORIA_MAP.keys()))
        genero    = st.selectbox("Género musical", list(GENERO_MAP.keys()))
        presupuesto = st.number_input("Presupuesto ($)", min_value=0, max_value=50_000_000,
                                       value=5_000_000, step=500_000)
        descuento = st.slider("Descuento aplicado (%)", 0.0, 0.5, 0.10, 0.05)

    with col_f2:
        st.markdown("**Condiciones del evento**")
        clima    = st.selectbox("Clima esperado", list(CLIMA_MAP.keys()))
        dia_sem  = st.selectbox("Día de la semana",
                                ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
        mes      = st.selectbox("Mes del evento",
                                ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                                 "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])
        es_festivo  = st.toggle("¿Día festivo?", value=False)
        es_dia_pago = st.toggle("¿Día de pago / quincena?", value=False)

    with col_f3:
        st.markdown("**Operación y logística**")
        n_personal = st.number_input("Personal asignado (personas)", min_value=1, max_value=50, value=8)
        preventa   = st.number_input("Entradas en preventa (unidades)", min_value=0,
                                      max_value=5000, value=300, step=50)

        # Contexto histórico similar
        cat_stats = df[df["categoria"] == categoria]
        if len(cat_stats) > 0:
            st.markdown(f"""
            <div class="risk-card" style="margin-top:.8rem">
              <div style="font-size:.68rem;color:#C9A227;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem">
                Histórico — {categoria}
              </div>
              <div style="font-size:.8rem;color:#C9D1D9;line-height:1.8">
                Asistencia prom.: <b>{cat_stats['asistencia_total'].mean():.0f}</b><br>
                Tasa ocup. prom.: <b>{cat_stats['tasa_ocupacion'].mean():.1f}%</b><br>
                Ingreso prom.: <b>${cat_stats['ingreso_total'].mean()/1e6:.1f}M</b><br>
                % Aforo crítico: <b>{cat_stats['aforo_critico'].mean()*100:.0f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🎯 Generar predicción y prescripción", type="primary"):
        dia_num = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"].index(dia_sem)
        mes_num = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                   "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"].index(mes) + 1

        vals = {
            "categoria_enc":    CATEGORIA_MAP.get(categoria, 0),
            "genero_enc":       GENERO_MAP.get(genero, 6),
            "clima_enc":        CLIMA_MAP.get(clima, 3),
            "es_festivo":       int(es_festivo),
            "es_dia_pago":      int(es_dia_pago),
            "dia_semana":       dia_num,
            "mes":              mes_num,
            "presupuesto":      presupuesto,
            "descuento":        descuento,
            "n_personal":       n_personal,
            "entradas_preventa":preventa,
        }

        asist_pred, prob_pct, presc = modelo.predecir(vals)
        nivel  = presc["nivel"]
        color  = presc["color"]
        staff  = presc["staffing_recomendado"]

        # ── Resultado principal
        st.markdown("---")
        st.markdown("#### Resultado de la predicción")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Asistencia estimada", f"{asist_pred:,.0f} personas")
        r2.metric("Probabilidad aforo crítico", f"{prob_pct:.1f}%")
        r3.metric("Personal recomendado", f"{staff} personas")
        r4.metric("Clasificación", AFORO_LABELS.get(1 if prob_pct >= 50 else 0, ""))

        # ── Medidor de probabilidad
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            delta={"reference": 50, "valueformat": ".1f",
                   "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#3FB950"}},
            title={"text": "Probabilidad de Aforo Crítico (%)", "font": {"color": "#C9D1D9", "size": 14}},
            number={"font": {"color": color, "size": 48}, "suffix": "%"},
            gauge={
                "axis":     {"range": [0, 100], "tickcolor": "#8B949E",
                             "tickfont": {"color": "#8B949E"}},
                "bar":      {"color": color, "thickness": 0.25},
                "bgcolor":  "#12121C",
                "bordercolor": "#1E1E30",
                "steps": [
                    {"range": [0,  40], "color": "rgba(63,185,80,.15)"},
                    {"range": [40, 85], "color": "rgba(201,162,39,.15)"},
                    {"range": [85,100], "color": "rgba(239,68,68,.15)"},
                ],
                "threshold": {"line": {"color": "#EF4444","width": 3},
                              "thickness": 0.8, "value": UMBRAL_CRITICO},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#12121C", font=dict(color="#8B949E"),
            margin=dict(l=30, r=30, t=60, b=20), height=280,
        )

        col_g, col_p = st.columns([1.2, 1.8])
        with col_g:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_p:
            border = color
            bg_map = {"critico": "rgba(239,68,68,.08)", "moderado": "rgba(201,162,39,.08)",
                      "bajo": "rgba(63,185,80,.08)"}
            icon_map = {"critico": "🔴", "moderado": "🟡", "bajo": "🟢"}
            titulo_map = {"critico": "AFORO CRÍTICO — Acción urgente requerida",
                          "moderado": "AFORO MODERADO — Monitoreo activo",
                          "bajo": "AFORO BAJO — Activar estrategia de demanda"}

            st.markdown(f"""
            <div style="background:{bg_map[nivel]};border:1px solid {border};border-left:4px solid {border};
                        border-radius:10px;padding:1.25rem;height:100%">
              <div style="font-size:.75rem;color:{border};font-weight:700;text-transform:uppercase;
                          letter-spacing:.07em;margin-bottom:.75rem">
                {icon_map[nivel]} &nbsp;{titulo_map[nivel]}
              </div>
            """, unsafe_allow_html=True)

            for i, accion in enumerate(presc["acciones"]):
                st.markdown(f"""
                <div class="action-item" style="background:rgba(0,0,0,.2);border-color:{border};color:#C9D1D9">
                  <span style="color:{border};font-weight:600">{i+1}.</span> {accion}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
              <div style="margin-top:.75rem;font-size:.8rem;color:#8B949E">
                Staff recomendado: <b style="color:{border}">{staff} personas</b>
                &nbsp;·&nbsp; Probabilidad: <b style="color:{border}">{prob_pct:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Comparativa con histórico similar
        similares = df[(df["categoria"] == categoria) & (df["clima"] == clima)]
        if len(similares) > 0:
            st.markdown("##### Eventos similares en el histórico")
            cols_sim = ["nombre","asistencia_total","tasa_ocupacion","ingreso_total","aforo_critico"]
            df_sim = similares[cols_sim].copy()
            df_sim.columns = ["Evento","Asistencia real","Tasa ocup. (%)","Ingreso ($)","Crítico"]
            df_sim["Crítico"] = df_sim["Crítico"].map({0:"No","No":0,1:"Sí","Sí":1})
            st.dataframe(df_sim.style.format({
                "Asistencia real": "{:,.0f}", "Tasa ocup. (%)": "{:.1f}",
                "Ingreso ($)": "${:,.0f}",
            }), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════
# TAB 2: Ranking de eventos
# ═════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("#### Ranking de todos los eventos por probabilidad de aforo crítico")
    st.markdown("""
    <div class="info-box">
      📋 Todos los eventos del histórico clasificados por su probabilidad predicha de alcanzar
      aforo crítico (≥85% ocupación). Usa esta tabla para priorizar recursos y activar
      protocolos con anticipación.
    </div>
    """, unsafe_allow_html=True)

    df_rank = modelo.predecir_dataset_completo()
    if df_rank is not None:
        cols_rank = ["nombre","categoria","clima","dia_semana","asistencia_total",
                     "tasa_ocupacion","prob_critico_pred","aforo_pred_label","nivel_prescripcion","ingreso_total"]
        df_rank_show = df_rank[cols_rank].copy()
        DIAS = {0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"}
        df_rank_show["dia_semana"] = df_rank_show["dia_semana"].map(DIAS)
        df_rank_show = df_rank_show.sort_values("prob_critico_pred", ascending=False)
        df_rank_show.columns = ["Evento","Categoría","Clima","Día","Asistencia real",
                                 "Ocup. (%)","Prob. crítico (%)","Clasificación","Nivel","Ingreso ($)"]

        # Filtros
        fc1, fc2 = st.columns(2)
        with fc1:
            nivel_f = st.multiselect("Filtrar por nivel",
                                     ["Crítico","Moderado","Bajo"],
                                     default=["Crítico","Moderado","Bajo"])
        with fc2:
            cat_f = st.multiselect("Filtrar por categoría",
                                   sorted(df_rank_show["Categoría"].unique()),
                                   default=sorted(df_rank_show["Categoría"].unique()))

        df_rank_f = df_rank_show[
            df_rank_show["Nivel"].isin(nivel_f) &
            df_rank_show["Categoría"].isin(cat_f)
        ]

        k1, k2, k3 = st.columns(3)
        k1.metric("Eventos mostrados", len(df_rank_f))
        k2.metric("Prob. crítico promedio", f"{df_rank_f['Prob. crítico (%)'].mean():.1f}%")
        k3.metric("Ingreso promedio", f"${df_rank_f['Ingreso ($)'].mean()/1e6:.1f}M")

        def color_prob(val):
            if isinstance(val, (float, int)):
                if val >= 85: return "color:#EF4444;font-weight:600"
                if val >= 40: return "color:#C9A227"
                return "color:#3FB950"
            return ""

        st.dataframe(
            df_rank_f.style.map(color_prob, subset=["Prob. crítico (%)"]).format({
                "Asistencia real": "{:,.0f}", "Ocup. (%)": "{:.1f}",
                "Prob. crítico (%)": "{:.1f}", "Ingreso ($)": "${:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=480,
        )

        # Exportar
        csv = df_rank_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar ranking CSV", csv,
                           "ranking_prescripcion_monastery.csv", "text/csv")

# ═════════════════════════════════════════════════════════
# TAB 3: Lealtad y Valor del Cliente (CLV)
# ═════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("#### Análisis de Lealtad y Valor por Categoría de Evento")
    st.markdown("""
    <div class="info-box">
      💎 &nbsp;<b>Valor del cliente (CLV proxy):</b> Se estima el valor acumulado generado por cada
      categoría de evento, combinando ingreso total, frecuencia y consistencia de ocupación.
      Esto permite priorizar categorías con mayor retorno y diseñar estrategias de lealtad
      (beneficios para clientes frecuentes, preventa exclusiva, etc.).
    </div>
    """, unsafe_allow_html=True)

    # CLV por categoría
    clv = df.groupby("categoria").agg(
        n_eventos        =("id_evento","count"),
        ingreso_total    =("ingreso_total","sum"),
        ingreso_prom     =("ingreso_total","mean"),
        ocupacion_prom   =("tasa_ocupacion","mean"),
        ocupacion_std    =("tasa_ocupacion","std"),
        asistencia_prom  =("asistencia_total","mean"),
        pct_critico      =("aforo_critico","mean"),
        consumo_por_asist=("consumo_por_asistente","mean"),
    ).reset_index()

    # Score de lealtad: ingreso_prom * ocupacion_prom / 100 * consistencia
    clv["consistencia"]   = 1 - (clv["ocupacion_std"].fillna(0) / 100)
    clv["clv_score"]      = (
        clv["ingreso_prom"] * clv["ocupacion_prom"] / 100 * clv["consistencia"]
    ).round(0)
    clv["clv_score_norm"] = (clv["clv_score"] / clv["clv_score"].max() * 100).round(1)
    clv = clv.sort_values("clv_score_norm", ascending=False)

    # Segmentar en 3 niveles
    def segmento(score):
        if score >= 66: return "🥇 Gold — Alta prioridad"
        if score >= 33: return "🥈 Silver — Prioridad media"
        return "🥉 Bronze — Activar crecimiento"

    clv["segmento"] = clv["clv_score_norm"].apply(segmento)

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        PAL = ["#C9A227","#3FB950","#58A6FF","#8B5CF6","#EF4444",
               "#F97316","#06B6D4","#EC4899","#84CC16","#A78BFA"]
        fig_clv = go.Figure(go.Bar(
            x=clv["clv_score_norm"], y=clv["categoria"], orientation="h",
            marker_color=PAL[:len(clv)],
            text=[f"{v:.0f}" for v in clv["clv_score_norm"]],
            textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
            customdata=clv[["n_eventos","ingreso_prom","ocupacion_prom","segmento"]].values,
            hovertemplate="<b>%{y}</b><br>CLV Score: %{x:.0f}/100<br>N° eventos: %{customdata[0]}<br>Ingreso prom: $%{customdata[1]:,.0f}<br>Ocup. prom: %{customdata[2]:.1f}%<br>Segmento: %{customdata[3]}<extra></extra>",
        ))
        fig_clv.update_layout(
            title=dict(text="CLV Score por categoría (0–100)", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D0D14", paper_bgcolor="#12121C", font=dict(color="#8B949E", size=11),
            xaxis=dict(gridcolor="#1E1E30", range=[0,115], title="CLV Score", tickfont=dict(color="#8B949E")),
            yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
            margin=dict(l=10, r=60, t=50, b=30), height=380,
        )
        st.plotly_chart(fig_clv, use_container_width=True)

    with col_l2:
        # Radar de atributos por segmento
        gold   = clv[clv["clv_score_norm"] >= 66]
        silver = clv[(clv["clv_score_norm"] >= 33) & (clv["clv_score_norm"] < 66)]
        bronze = clv[clv["clv_score_norm"] < 33]

        attrs = ["Ingreso prom. (norm)", "Ocupación prom.", "Consistencia",
                 "% Aforo crítico", "Consumo/asist. (norm)"]

        def norm(series):
            mx = series.max()
            return (series / mx * 100).mean() if mx > 0 else 0

        def radar_vals(subset):
            if len(subset) == 0:
                return [0]*5
            return [
                norm(subset["ingreso_prom"]),
                subset["ocupacion_prom"].mean(),
                subset["consistencia"].mean() * 100,
                subset["pct_critico"].mean() * 100,
                norm(subset["consumo_por_asist"]),
            ]

        fig_radar = go.Figure()
        for seg, color, vals_seg in [
            ("Gold",   "#C9A227", radar_vals(gold)),
            ("Silver", "#8B949E", radar_vals(silver)),
            ("Bronze", "#CD7F32", radar_vals(bronze)),
        ]:
            if any(v > 0 for v in vals_seg):
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_seg + [vals_seg[0]], theta=attrs + [attrs[0]],
                    name=seg, line_color=color, fill="toself",
                    fillcolor=color.replace("FF","18").replace("27","18").replace("49","18"),
                    opacity=0.85,
                ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#12121C",
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor="#1E1E30", tickfont=dict(color="#8B949E", size=9)),
                angularaxis=dict(tickfont=dict(color="#C9D1D9", size=10), gridcolor="#1E1E30"),
            ),
            paper_bgcolor="#12121C",
            legend=dict(bgcolor="#12121C", bordercolor="#1E1E30", font=dict(color="#C9D1D9", size=10)),
            title=dict(text="Radar de atributos por segmento", font=dict(color="#C9D1D9", size=13)),
            margin=dict(l=40, r=40, t=60, b=40), height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tabla de segmentos con prescripción
    st.markdown("##### Estrategia prescriptiva por segmento")
    ESTRATEGIAS = {
        "🥇 Gold — Alta prioridad": {
            "color": "#C9A227",
            "acciones": [
                "Programar con mayor frecuencia — son los eventos de mayor retorno.",
                "Ofrecer acceso exclusivo a preventa VIP para clientes recurrentes.",
                "Aumentar inversión en marketing digital 2 semanas antes.",
                "Asegurar DJ o artista de alto perfil para mantener la expectativa.",
                "Activar sistema de lista de espera para eventos sold-out.",
            ],
        },
        "🥈 Silver — Prioridad media": {
            "color": "#8B949E",
            "acciones": [
                "Evaluar ajuste de precio de cover para maximizar el margen.",
                "Implementar paquetes grupales (4+ personas) con descuento.",
                "Medir el impacto real del DJ invitado vs costo adicional.",
                "Incrementar activaciones en redes 1 semana antes.",
                "Analizar si el día de semana es el óptimo para esta categoría.",
            ],
        },
        "🥉 Bronze — Activar crecimiento": {
            "color": "#CD7F32",
            "acciones": [
                "Revisar el concepto del evento — puede no conectar con el mercado local.",
                "Probar alianzas con marcas externas para inyectar audiencia nueva.",
                "Ofrecer entrada gratuita para los primeros 50 asistentes (efecto arrastre).",
                "Evaluar fusionar esta categoría con una Gold para reducir costos.",
                "Rediseñar la propuesta de valor del evento (temática, horario, precio).",
            ],
        },
    }

    for seg_nombre, info in ESTRATEGIAS.items():
        seg_data = clv[clv["segmento"] == seg_nombre]
        if len(seg_data) == 0:
            continue
        categorias_seg = ", ".join(seg_data["categoria"].tolist())
        with st.expander(f"{seg_nombre} — {categorias_seg}", expanded=(seg_nombre.startswith("🥇"))):
            col_e1, col_e2 = st.columns([1, 2])
            with col_e1:
                st.markdown(f"""
                <div class="risk-card">
                  <div style="font-size:.7rem;color:{info['color']};font-weight:600;text-transform:uppercase;
                              letter-spacing:.06em;margin-bottom:.6rem">Métricas del segmento</div>
                  <div style="font-size:.82rem;color:#C9D1D9;line-height:1.9">
                    Categorías: <b>{len(seg_data)}</b><br>
                    CLV Score prom.: <b>{seg_data['clv_score_norm'].mean():.0f}/100</b><br>
                    Ingreso prom.: <b>${seg_data['ingreso_prom'].mean()/1e6:.1f}M</b><br>
                    Ocupación prom.: <b>{seg_data['ocupacion_prom'].mean():.1f}%</b><br>
                    % Aforo crítico: <b>{seg_data['pct_critico'].mean()*100:.0f}%</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with col_e2:
                st.markdown(f"<div style='font-size:.75rem;color:{info['color']};font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem'>Acciones prescriptivas</div>", unsafe_allow_html=True)
                for i, accion in enumerate(info["acciones"]):
                    st.markdown(f"""
                    <div class="action-item" style="background:rgba(0,0,0,.2);border-color:{info['color']};color:#C9D1D9">
                      <span style="color:{info['color']};font-weight:600">{i+1}.</span> {accion}
                    </div>
                    """, unsafe_allow_html=True)
