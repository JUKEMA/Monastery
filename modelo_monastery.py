"""
modelo_monastery.py
MONASTERY Club · Analítica Predictiva de Aforo y Demanda
Business Analytics 801 · UDeC · 2026
Encapsula CRISP-DM Fases 3-5: Preparación, Modelado y Evaluación.
Dos capas de predicción:
  · Regresión   → asistencia_total (número de asistentes)
  · Clasificación → aforo_critico  (evento ≥ 85% ocupación: Sí/No)
"""

import numpy as np
import pandas as pd

# ── Catálogos de encoding
CLIMA_MAP = {
    "Soleado": 6, "Despejado": 5, "Parcial": 4,
    "Nublado": 3, "Ventoso": 2, "Lluvioso": 1, "Tormentoso": 0,
}
CATEGORIA_MAP = {
    "Concierto": 0, "Festival": 1, "Feria": 2, "Exposicion": 3,
    "Deportivo": 4, "Academico": 5, "Gastronomico": 6,
    "Infantil": 7, "Conferencia": 8, "Obra_Teatro": 9,
}
GENERO_MAP = {
    "Rock": 0, "Pop": 1, "Reggaeton": 2, "Electronica": 3,
    "Crossover": 4, "Tecnologia": 5, "General": 6,
    "Clasica": 7, "Jazz": 8, "Urbano": 9,
}

FEATURE_COLS = [
    "categoria_enc", "genero_enc", "clima_enc",
    "es_festivo", "es_dia_pago", "dia_semana", "mes",
    "presupuesto", "descuento", "n_personal",
    "entradas_preventa",
]

FEATURE_DISPLAY = {
    "categoria_enc":    "Categoría del evento",
    "genero_enc":       "Género musical",
    "clima_enc":        "Condición climática",
    "es_festivo":       "Día festivo",
    "es_dia_pago":      "Día de pago/quincena",
    "dia_semana":       "Día de la semana",
    "mes":              "Mes del año",
    "presupuesto":      "Presupuesto evento ($)",
    "descuento":        "Descuento aplicado (%)",
    "n_personal":       "Personal asignado",
    "entradas_preventa":"Entradas en preventa",
}

AFORO_LABELS = {0: "Aforo Normal", 1: "Aforo Crítico"}
AFORO_COLORS = {0: "#3FB950", 1: "#EF4444"}
UMBRAL_CRITICO = 85.0   # % de ocupación → aforo crítico


class ModeloMonastery:
    """
    Encapsula CRISP-DM Fases 3–5 para MONASTERY Club.
    Modelo de regresión  : predice asistencia_total (número de personas).
    Modelo de clasificación: predice aforo_critico (≥85% capacidad).
    """

    def __init__(self):
        self.trained          = False
        self.modelo_reg       = {}   # {nombre: objeto sklearn}
        self.modelo_clf       = {}
        self.resultados_reg   = {}   # métricas de regresión por modelo
        self.resultados_clf   = {}   # métricas de clasificación por modelo
        self.importancia_reg  = {}
        self.importancia_clf  = {}
        self.df_master        = None  # dataset maestro unificado
        self.df_prepared      = None  # dataset con features listo
        self.scaler_reg       = None
        self.scaler_clf       = None
        self.feature_names    = FEATURE_COLS.copy()
        self.n_train          = 0
        self.n_test           = 0
        self.stats            = {}

    # ══════════════════════════════════════════════════════════════
    # CRISP-DM Fase 2-3: Comprensión y Preparación de Datos
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def cargar_excel(path_or_file) -> pd.DataFrame:
        """
        Lee el Excel de MONASTERY y construye el dataset maestro
        unificando las 9 tablas del esquema estrella.
        Retorna un DataFrame plano con todas las features.
        """
        wb_data = pd.ExcelFile(path_or_file)

        def leer(nombre):
            return pd.read_excel(wb_data, sheet_name=nombre)

        cal   = leer("DIM_CALENDARIO")
        tipo  = leer("DIM_TIPO_EVENTO")
        piv_c = leer("PIVOTE_EVENTO_CALENDARIO")
        piv_p = leer("PIVOTE_EVENTO_PERSONAL")
        fact  = leer("FACT_EVENTOS")
        vtas  = leer("FACT_VENTAS")
        asist = leer("FACT_ASISTENCIA")

        # Personal por evento
        pers_cnt = piv_p.groupby("id_evento").size().reset_index(name="n_personal")

        # Clima y calendario por evento (tomar el primer registro por evento)
        clima_ev = (
            piv_c
            .merge(cal, on="id_clima", how="left")
            .groupby("id_evento")
            .agg(clima=("clima", "first"),
                 es_festivo=("es_festivo", "max"),
                 es_dia_pago=("es_dia_pago", "max"))
            .reset_index()
        )

        # Merge maestro
        df = (fact
              .merge(tipo,    on="id_tipo",              how="left")
              .merge(vtas,    on="id_evento",             how="left")
              .merge(asist,   on=["id_evento", "id_venta"], how="left")
              .merge(clima_ev, on="id_evento",            how="left")
              .merge(pers_cnt, on="id_evento",            how="left"))

        df["fecha"]      = pd.to_datetime(df["fecha"], errors="coerce")
        df["dia_semana"] = df["fecha"].dt.dayofweek        # 0=Lun … 6=Dom
        df["mes"]        = df["fecha"].dt.month

        # KPIs derivados
        df["tasa_ocupacion"] = (
            df["asistencia_total"] / df["aforo_maximo"].replace(0, np.nan) * 100
        ).round(2)
        df["ingreso_total"] = (
            df["consumo_bebidas"]
            + df["entradas_preventa"]  * 25_000
            + df["entradas_taquilla"]  * 30_000
        ).round(0)
        df["consumo_por_asistente"] = (
            df["consumo_bebidas"] / df["asistencia_total"].replace(0, np.nan)
        ).round(0)
        df["aforo_critico"] = (df["tasa_ocupacion"] >= UMBRAL_CRITICO).astype(int)
        df["ratio_preventa"] = (
            df["entradas_preventa"] / (
                df["entradas_preventa"] + df["entradas_taquilla"]
            ).replace(0, np.nan)
        ).round(3)
        df["eficiencia_personal"] = (
            df["asistencia_total"] / df["n_personal"].replace(0, np.nan)
        ).round(1)

        # Aforo máximo teórico (máximo histórico por categoría)
        aforo_max_teorico = (
            df.groupby("categoria")["asistencia_total"].max()
            .reset_index(name="aforo_max_teorico")
        )
        df = df.merge(aforo_max_teorico, on="categoria", how="left")
        df["brecha_aforo"] = df["aforo_max_teorico"] - df["asistencia_total"]
        df["pct_gap"] = (
            df["brecha_aforo"] / df["aforo_max_teorico"] * 100
        ).round(1)

        return df

    def preparar(self, df_master: pd.DataFrame):
        """
        CRISP-DM Fase 3: encoding, imputación, construcción de features.
        Retorna X, y_reg, y_clf, feature_names.
        """
        df = df_master.copy()

        # Label Encoding
        df["categoria_enc"] = df["categoria"].map(CATEGORIA_MAP).fillna(0).astype(int)
        df["genero_enc"]    = df["genero_principal"].map(GENERO_MAP).fillna(6).astype(int)
        df["clima_enc"]     = df["clima"].map(CLIMA_MAP).fillna(3).astype(int)

        # Imputar nulos con mediana
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Variables objetivo
        y_reg = df["asistencia_total"].values.astype(float)
        y_clf = df["aforo_critico"].values.astype(int)

        # Calcular estadísticas EDA
        self.stats = self._calc_stats(df)
        self.df_prepared = df.copy()

        X = df[FEATURE_COLS].values
        return X, y_reg, y_clf, FEATURE_COLS

    def _calc_stats(self, df):
        """Estadísticas descriptivas para el dashboard."""
        corr_cols = [
            "entradas_preventa", "n_personal", "presupuesto",
            "es_festivo", "clima_enc", "dia_semana",
        ]
        corr = {}
        for c in corr_cols:
            if c in df.columns:
                try:
                    v_c = df[c].fillna(0).values.astype(float)
                    v_t = df["asistencia_total"].fillna(0).values.astype(float)
                    mx, mt = v_c.mean(), v_t.mean()
                    num = ((v_c - mx) * (v_t - mt)).sum()
                    dx  = np.sqrt(((v_c - mx)**2).sum() or 1)
                    dt  = np.sqrt(((v_t - mt)**2).sum() or 1)
                    corr[c] = round(float(num / (dx * dt)), 3)
                except Exception:
                    corr[c] = 0.0

        return {
            "n_total":             len(df),
            "n_critico":           int(df["aforo_critico"].sum()),
            "n_normal":            int((df["aforo_critico"] == 0).sum()),
            "pct_critico":         round(df["aforo_critico"].mean() * 100, 1),
            "avg_asistencia":      round(df["asistencia_total"].mean(), 0),
            "avg_tasa_ocupacion":  round(df["tasa_ocupacion"].mean(), 1),
            "avg_ingreso":         round(df["ingreso_total"].mean(), 0),
            "avg_consumo_asist":   round(df["consumo_por_asistente"].mean(), 0),
            "avg_personal":        round(df["n_personal"].mean(), 1),
            "max_asistencia":      int(df["asistencia_total"].max()),
            "min_asistencia":      int(df["asistencia_total"].min()),
            "total_ingresos":      int(df["ingreso_total"].sum()),
            "categoria_counts":    df["categoria"].value_counts().to_dict(),
            "clima_counts":        df["clima"].value_counts().to_dict(),
            "dia_counts":          df["dia_semana"].value_counts().sort_index().to_dict(),
            "mes_counts":          df["mes"].value_counts().sort_index().to_dict(),
            "correlaciones":       corr,
            "aforo_por_categoria": df.groupby("categoria")["tasa_ocupacion"].mean().round(1).to_dict(),
            "ingreso_por_categoria": df.groupby("categoria")["ingreso_total"].mean().round(0).to_dict(),
            "asistencia_por_clima": df.groupby("clima")["asistencia_total"].mean().round(0).to_dict(),
            "gap_por_categoria":   df.groupby("categoria")["pct_gap"].mean().round(1).to_dict(),
        }

    # ══════════════════════════════════════════════════════════════
    # CRISP-DM Fase 4: Modelado
    # ══════════════════════════════════════════════════════════════
    def entrenar(self, df_master: pd.DataFrame, progress_cb=None):
        """
        Entrena 3 modelos de regresión y 3 de clasificación.
        Retorna (resultados_reg, resultados_clf).
        """
        from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix, roc_auc_score,
        )

        try:
            from xgboost import XGBRegressor, XGBClassifier
            tiene_xgb = True
        except ImportError:
            tiene_xgb = False

        if progress_cb: progress_cb(5, "Preparando datos...")
        X, y_reg, y_clf, feature_names = self.preparar(df_master)
        self.feature_names = feature_names
        self.df_master     = df_master.copy()

        # Split estratificado (usando aforo_critico para mantener balance)
        X_tr, X_te, y_r_tr, y_r_te, y_c_tr, y_c_te = train_test_split(
            X, y_reg, y_clf,
            test_size=0.2, random_state=42, stratify=y_clf,
        )
        self.n_train = len(X_tr)
        self.n_test  = len(X_te)

        # Escaladores
        sc_r = StandardScaler(); sc_c = StandardScaler()
        X_tr_sr = sc_r.fit_transform(X_tr); X_te_sr = sc_r.transform(X_te)
        X_tr_sc = sc_c.fit_transform(X_tr); X_te_sc = sc_c.transform(X_te)
        self.scaler_reg = sc_r; self.scaler_clf = sc_c

        # ── Modelos de regresión
        if progress_cb: progress_cb(15, "Entrenando modelos de regresión...")
        modelos_reg = {
            "Regresión Lineal": LinearRegression(),
            "Random Forest Reg.": RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        }
        if tiene_xgb:
            modelos_reg["XGBoost Reg."] = XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0, n_jobs=-1,
            )
        else:
            # HistGradientBoosting: versión rápida de sklearn (~10x más veloz que GradientBoosting)
            modelos_reg["Hist Gradient Boosting"] = HistGradientBoostingRegressor(
                max_iter=100, max_depth=5, learning_rate=0.1,
                random_state=42, early_stopping=False)

        cv_r = KFold(n_splits=5, shuffle=True, random_state=42)
        res_reg = {}
        for idx, (nombre, modelo) in enumerate(modelos_reg.items()):
            pct = 15 + int(25 * idx / len(modelos_reg))
            if progress_cb: progress_cb(pct, f"Entrenando {nombre}...")
            if nombre == "Regresión Lineal":
                modelo.fit(X_tr_sr, y_r_tr)
                y_pred = modelo.predict(X_te_sr)
                cv_sc  = cross_val_score(modelo, sc_r.transform(X), y_reg, cv=cv_r,
                                         scoring="r2", n_jobs=-1)
            else:
                modelo.fit(X_tr, y_r_tr)
                y_pred = modelo.predict(X_te)
                cv_sc  = cross_val_score(modelo, X, y_reg, cv=cv_r,
                                         scoring="r2", n_jobs=-1)

            y_pred = np.clip(y_pred, 0, None)
            rmse   = float(np.sqrt(mean_squared_error(y_r_te, y_pred)))
            mae    = float(mean_absolute_error(y_r_te, y_pred))
            r2     = float(r2_score(y_r_te, y_pred))

            res_reg[nombre] = {
                "rmse":    round(rmse, 2),
                "mae":     round(mae, 2),
                "r2":      round(r2, 4),
                "cv_mean": round(float(cv_sc.mean()), 4),
                "cv_std":  round(float(cv_sc.std()), 4),
                "y_test":  y_r_te.tolist(),
                "y_pred":  y_pred.tolist(),
                "modelo":  modelo,
            }

        # ── Modelos de clasificación
        if progress_cb: progress_cb(45, "Entrenando modelos de clasificación...")
        modelos_clf = {
            "Regresión Logística": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, class_weight="balanced"),
            "Random Forest Clf.": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42,
                n_jobs=-1, class_weight="balanced"),
        }
        if tiene_xgb:
            modelos_clf["XGBoost Clf."] = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1,
                scale_pos_weight=float((y_clf == 0).sum()) / ((y_clf == 1).sum() + 1),
            )

        cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        res_clf = {}
        for idx, (nombre, modelo) in enumerate(modelos_clf.items()):
            pct = 50 + int(30 * idx / len(modelos_clf))
            if progress_cb: progress_cb(pct, f"Entrenando {nombre}...")
            if nombre == "Regresión Logística":
                modelo.fit(X_tr_sc, y_c_tr)
                y_pred  = modelo.predict(X_te_sc)
                y_proba = modelo.predict_proba(X_te_sc)[:, 1]
                cv_sc   = cross_val_score(modelo, sc_c.transform(X), y_clf, cv=cv_c,
                                          scoring="f1", n_jobs=-1)
            else:
                modelo.fit(X_tr, y_c_tr)
                y_pred  = modelo.predict(X_te)
                y_proba = modelo.predict_proba(X_te)[:, 1]
                cv_sc   = cross_val_score(modelo, X, y_clf, cv=cv_c,
                                          scoring="f1", n_jobs=-1)

            cm = confusion_matrix(y_c_te, y_pred, labels=[0, 1])
            try:
                auc = round(float(roc_auc_score(y_c_te, y_proba)), 4)
            except Exception:
                auc = None

            res_clf[nombre] = {
                "accuracy":  round(float(accuracy_score(y_c_te, y_pred)), 4),
                "precision": round(float(precision_score(y_c_te, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_c_te, y_pred, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_c_te, y_pred, zero_division=0)), 4),
                "roc_auc":   auc,
                "cm":        cm.tolist(),
                "cv_mean":   round(float(cv_sc.mean()), 4),
                "cv_std":    round(float(cv_sc.std()), 4),
                "y_test":    y_c_te.tolist(),
                "y_proba":   y_proba.tolist(),
                "modelo":    modelo,
            }

        # ── Importancia de variables (RF)
        if progress_cb: progress_cb(85, "Calculando importancia de variables...")
        rf_r = res_reg.get("Random Forest Reg.", {}).get("modelo")
        if rf_r and hasattr(rf_r, "feature_importances_"):
            self.importancia_reg = dict(sorted(
                {fn: round(float(v), 4) for fn, v in
                 zip(feature_names, rf_r.feature_importances_)}.items(),
                key=lambda x: x[1], reverse=True,
            ))
        rf_c = res_clf.get("Random Forest Clf.", {}).get("modelo")
        if rf_c and hasattr(rf_c, "feature_importances_"):
            self.importancia_clf = dict(sorted(
                {fn: round(float(v), 4) for fn, v in
                 zip(feature_names, rf_c.feature_importances_)}.items(),
                key=lambda x: x[1], reverse=True,
            ))

        # Guardar sin el objeto modelo (para serialización)
        self.resultados_reg = {
            k: {kk: vv for kk, vv in v.items() if kk != "modelo"}
            for k, v in res_reg.items()
        }
        self.resultados_clf = {
            k: {kk: vv for kk, vv in v.items() if kk != "modelo"}
            for k, v in res_clf.items()
        }
        self.modelo_reg = {k: v["modelo"] for k, v in res_reg.items()}
        self.modelo_clf = {k: v["modelo"] for k, v in res_clf.items()}
        self.trained = True

        if progress_cb: progress_cb(100, "¡Entrenamiento completado!")
        return self.resultados_reg, self.resultados_clf

    # ══════════════════════════════════════════════════════════════
    # CRISP-DM Fase 5: Predicción individual
    # ══════════════════════════════════════════════════════════════
    def predecir(self, valores_dict: dict,
                 modelo_reg_nombre: str = "Random Forest Reg.",
                 modelo_clf_nombre: str = "Random Forest Clf."):
        """
        Predicción individual para un evento.
        Retorna (asistencia_pred, prob_critico, prescripcion).
        """
        if not self.trained:
            raise RuntimeError("Modelo no entrenado.")
        x = np.array([float(valores_dict.get(f, 0.0))
                      for f in self.feature_names]).reshape(1, -1)

        # Regresión
        mod_r = self.modelo_reg[modelo_reg_nombre]
        if modelo_reg_nombre == "Regresión Lineal":
            x_r = self.scaler_reg.transform(x)
        else:
            x_r = x
        asistencia_pred = max(0.0, float(mod_r.predict(x_r)[0]))

        # Clasificación
        mod_c = self.modelo_clf[modelo_clf_nombre]
        if modelo_clf_nombre == "Regresión Logística":
            x_c = self.scaler_clf.transform(x)
        else:
            x_c = x
        prob_critico = float(mod_c.predict_proba(x_c)[0][1])

        # Prescripción automática
        prescripcion = self._prescribir(prob_critico, valores_dict)

        return round(asistencia_pred, 0), round(prob_critico * 100, 1), prescripcion

    def _prescribir(self, prob: float, vals: dict) -> dict:
        """Genera acciones prescriptivas según umbral de probabilidad."""
        if prob >= 0.85:
            nivel = "critico"
            color = "#EF4444"
            acciones = [
                "Reforzar equipo de seguridad (+2 vigilantes mínimo)",
                "Aumentar inventario de bebidas premium en un 30%",
                "Activar protocolos de control de ingreso (manillas, turnos)",
                "Coordinar con autoridades locales si supera 90% de aforo",
                "Notificar a logística para plan de contingencia de espacio",
            ]
            staffing = int(vals.get("n_personal", 5) * 1.4)
        elif prob >= 0.40:
            nivel = "moderado"
            color = "#F59E0B"
            acciones = [
                "Mantener staff estándar programado",
                "Gestión normal de inventario de bebidas",
                "Monitoreo regular cada 2 horas durante el evento",
                "Evaluar si activar publicidad digital adicional en redes",
                "Revisar preventa 48 horas antes para ajustar proyección",
            ]
            staffing = int(vals.get("n_personal", 5))
        else:
            nivel = "bajo"
            color = "#3FB950"
            acciones = [
                "Aumentar inversión en publicidad digital (Facebook/Instagram Ads)",
                "Activar promociones especiales o descuentos para grupos",
                "Evaluar cambio de DJ o género musical del evento",
                "Considerar descuento de preventa de último momento",
                "Revisar si la fecha compite con otro evento en la ciudad",
            ]
            staffing = max(3, int(vals.get("n_personal", 5) * 0.75))

        return {
            "nivel":    nivel,
            "color":    color,
            "acciones": acciones,
            "staffing_recomendado": staffing,
            "prob_pct": round(prob * 100, 1),
        }

    def predecir_dataset_completo(self, modelo_clf_nombre: str = "Random Forest Clf."):
        """Predice aforo_critico para todos los eventos y retorna df enriquecido."""
        if not self.trained or self.df_prepared is None:
            return None
        df = self.df_prepared.copy()
        X  = df[self.feature_names].values
        mod_c = self.modelo_clf[modelo_clf_nombre]
        if modelo_clf_nombre == "Regresión Logística":
            X = self.scaler_clf.transform(X)
        probas = mod_c.predict_proba(X)[:, 1]
        preds  = (probas >= 0.5).astype(int)
        df["prob_critico_pred"] = (probas * 100).round(1)
        df["aforo_pred_label"]  = [AFORO_LABELS[p] for p in preds]
        df["nivel_prescripcion"] = pd.cut(
            probas,
            bins=[0, 0.40, 0.85, 1.01],
            labels=["Bajo", "Moderado", "Crítico"],
            include_lowest=True,
        )
        return df

    def simular_escalamiento(self, factor_aforo: float = 2.0,
                              modelo_clf_nombre: str = "Random Forest Clf."):
        """
        Simula qué ocurre si se duplica (o modifica) la capacidad de aforo.
        Retorna df con métricas de cuello de botella.
        """
        if not self.trained or self.df_prepared is None:
            return None, {}
        df = self.df_prepared.copy()

        # Simular aumento de preventa proporcionalmente al aforo nuevo
        df_sim = df.copy()
        df_sim["entradas_preventa"] = (
            df_sim["entradas_preventa"] * factor_aforo
        ).round(0)
        df_sim["n_personal"] = (df_sim["n_personal"] * factor_aforo).round(0)

        X_sim = df_sim[self.feature_names].values
        mod_c = self.modelo_clf[modelo_clf_nombre]
        if modelo_clf_nombre == "Regresión Logística":
            X_sim = self.scaler_clf.transform(X_sim)
        probas_sim = mod_c.predict_proba(X_sim)[:, 1]

        df_sim["prob_critico_sim"] = (probas_sim * 100).round(1)
        df_sim["prob_critico_orig"] = (
            self.predecir_dataset_completo(modelo_clf_nombre)["prob_critico_pred"]
        )
        df_sim["delta_prob"] = (
            df_sim["prob_critico_sim"] - df_sim["prob_critico_orig"]
        ).round(1)

        cuellos = {
            "personal_critico": int(df_sim[df_sim["prob_critico_sim"] > 85]["n_personal"].mean() or 0),
            "pct_eventos_criticos_nuevo": round(float((probas_sim >= 0.85).mean() * 100), 1),
            "pct_eventos_criticos_orig":  round(float(df["aforo_critico"].mean() * 100), 1),
            "delta_criticos": round(
                float((probas_sim >= 0.85).mean() * 100) - float(df["aforo_critico"].mean() * 100), 1
            ),
            "ingreso_proyectado": int(df_sim["ingreso_total"].sum() * factor_aforo * 0.85),
            "ingreso_actual":     int(df["ingreso_total"].sum()),
        }
        return df_sim, cuellos
