import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
import io

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor IA · Segmentación",
    layout="wide",
    page_icon="🧬"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stat-box {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid #2a5f7f;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 6px 0;
    }
    .stat-label { font-size: 12px; color: #7ecfff; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { font-size: 26px; font-weight: 700; color: #ffffff; margin-top: 4px; }
    .stat-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
    .method-box {
        background: #1a1a2e;
        border-left: 4px solid #4ecdc4;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: monospace;
        font-size: 13px;
        color: #e0e0e0;
        line-height: 1.8;
    }
    .tag {
        display: inline-block;
        background: #2a5f7f;
        color: #7ecfff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PERFILES POR TIPO DE CLIENTE
# ─────────────────────────────────────────────────────────────────────────────
PERFILES = {
    "🏋️ Gimnasio": {
        "arquetipos": {
            "Fiel Activo":      ["siempre", "todos los días", "excelente", "recomiendo", "feliz"],
            "En Riesgo":        ["caro", "máquina", "rota", "sucio", "vestuario", "cancelar"],
            "Formativo":        ["clase", "profe", "instructor", "aprender", "pilates", "yoga"],
            "Social":           ["amigo", "grupo", "conocer", "compañero", "ambiente"],
            "Detractor":        ["malo", "pésimo", "nunca vuelvo", "decepcionado", "terrible"],
        },
        "col_sugeridas": ["segmento", "nota", "comentario", "frecuencia"]
    },
    "⚽ Club Deportivo": {
        "arquetipos": {
            "Competitivo":      ["torneo", "competir", "ganar", "copa", "medalla", "ranking"],
            "Social":           ["asado", "fiestas", "amigos", "familia", "compartir"],
            "Crítico Infra":    ["cancha", "baño", "luz", "agua", "vestuario", "infraestructura"],
            "Formativo":        ["taller", "escuela", "aprender", "técnica", "entrenamiento"],
            "Detractor":        ["malo", "pésimo", "caro", "desorganizado", "nunca vuelvo"],
        },
        "col_sugeridas": ["rama", "nota", "comentario", "socio"]
    },
    "🛒 Ecommerce": {
        "arquetipos": {
            "Cliente Fiel":     ["siempre", "compro seguido", "excelente", "recomiendo", "rápido"],
            "En Riesgo":        ["lento", "devolución", "problema", "tarde", "roto", "cancelar"],
            "Precio-Sensible":  ["caro", "descuento", "oferta", "precio", "económico"],
            "Experiencia":      ["fácil", "intuitivo", "web", "app", "diseño", "rápida"],
            "Detractor":        ["pésimo", "nunca más", "fraude", "estafa", "horrible"],
        },
        "col_sugeridas": ["segmento", "nps", "comentario", "ticket_promedio"]
    },
    "🏙️ Municipio / Comunidad": {
        "arquetipos": {
            "Crítico Infra":    ["luz", "agua", "basura", "calle", "bache", "semáforo"],
            "Participativo":    ["reunión", "juntar", "vecino", "participar", "propuesta"],
            "Insatisfecho":     ["malo", "nada", "nunca", "promesa", "abandono"],
            "Promotor":         ["bueno", "mejoró", "gracias", "excelente", "contento"],
            "Neutro":           [],
        },
        "col_sugeridas": ["sector", "nota", "comentario", "edad"]
    },
    "🏥 Salud / Clínica": {
        "arquetipos": {
            "Promotor":         ["excelente", "recomiendo", "atención", "rápido", "amable"],
            "En Espera":        ["espera", "hora", "demora", "turno", "lento"],
            "Crítico Trato":    ["malo", "grosero", "fría", "desatención", "olvidaron"],
            "Digital":          ["app", "online", "telemedicina", "virtual", "web"],
            "Detractor":        ["pésimo", "nunca vuelvo", "error", "equivocaron"],
        },
        "col_sugeridas": ["especialidad", "nota", "comentario", "edad"]
    },
    "⚙️ Personalizado": {
        "arquetipos": {},
        "col_sugeridas": []
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTODETECCIÓN DE COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "segmento": ["segmento", "rama", "sector", "categoria", "grupo", "plan", "tipo",
                 "especialidad", "categoria_cliente", "membership", "product"],
    "nps":      ["nps", "nota", "score", "puntuación", "rating", "calificacion",
                 "satisfaccion", "puntaje", "estrella", "stars", "note"],
    "texto":    ["comentario", "comentarios", "observacion", "texto", "opinion",
                 "feedback", "respuesta", "descripcion", "comment", "review"],
}

def similitud(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def autodetectar(columnas):
    resultado = {}
    for metric, aliases in COLUMN_ALIASES.items():
        mejor, mejor_score = None, 0
        for col in columnas:
            for alias in aliases:
                s = similitud(col, alias)
                if s > mejor_score and s >= 0.70:
                    mejor_score, mejor = s, col
        resultado[metric] = mejor
    return resultado

# ─────────────────────────────────────────────────────────────────────────────
# FÓRMULA MUESTRAL
# ─────────────────────────────────────────────────────────────────────────────
def calcular_muestra(N, Z, p, e):
    """Fórmula de Cochran con corrección para población finita."""
    n0 = (Z**2 * p * (1 - p)) / (e**2)
    n  = n0 / (1 + (n0 - 1) / N)
    return math.ceil(n)

# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN POR ARQUETIPOS
# ─────────────────────────────────────────────────────────────────────────────
def clasificar_fila(texto, nps_val, arquetipos_dict):
    txt = str(texto).lower()
    for arquetipo, keywords in arquetipos_dict.items():
        if keywords and any(kw in txt for kw in keywords):
            return arquetipo
    # Fallback por NPS
    try:
        score = float(nps_val)
        if score >= 7: return "Promotor"
        if score <= 4: return "Detractor"
        return "Neutro"
    except Exception:
        return "Neutro"

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE DATOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────
def generar_sinteticos(df_real, col_seg, col_arq, n_sinteticos, seed=42):
    np.random.seed(seed)
    dist_seg    = df_real[col_seg].value_counts(normalize=True)
    matrix_prob = pd.crosstab(df_real[col_seg], df_real[col_arq], normalize='index')
    global_prob = df_real[col_arq].value_counts(normalize=True)

    nuevos_seg = np.random.choice(dist_seg.index, size=n_sinteticos, p=dist_seg.values)
    rows = []
    for seg in nuevos_seg:
        if seg in matrix_prob.index:
            probs = matrix_prob.loc[seg]
            arq   = np.random.choice(probs.index, p=probs.values)
        else:
            arq = np.random.choice(global_prob.index, p=global_prob.values)
        rows.append({col_seg: seg, col_arq: arq, "Origen": "Sintético (IA)"})

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERIZACIÓN K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
def encontrar_k_optimo(X_scaled, k_max=8):
    inertias, silhouettes, ks = [], [], range(2, min(k_max + 1, len(X_scaled)))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    # K óptimo = mayor silhouette
    k_optimo = list(ks)[np.argmax(silhouettes)]
    return k_optimo, list(ks), inertias, silhouettes

def clusterizar(df, cols_numericas, k):
    X = df[cols_numericas].fillna(df[cols_numericas].median())
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = km.fit_predict(X_scaled).astype(str)
    return df, X_scaled

# ─────────────────────────────────────────────────────────────────────────────
# UI PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧬 Motor IA · Segmentación & Gemelos Digitales")
st.markdown("Herramienta universal para segmentar comunidades, completar muestras y clusterizar.")

# ══ TAB STRUCTURE ══
tab1, tab2, tab3 = st.tabs([
    "① Datos & Configuración",
    "② Muestra & Sintéticos",
    "③ Clusterización"
])

# ════════════════════════════════════════════════════════
# TAB 1 — DATOS & CONFIGURACIÓN
# ════════════════════════════════════════════════════════
with tab1:

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Tipo de Cliente")
        tipo_cliente = st.selectbox("Selecciona el perfil", list(PERFILES.keys()))
        perfil = PERFILES[tipo_cliente]

    with col_right:
        st.subheader("Cargar Datos")
        uploaded = st.file_uploader("Sube tu Excel o CSV", type=["xlsx", "xls", "csv"])

    if not uploaded:
        st.info("⬆️ Sube un archivo para continuar.")
        st.stop()

    # Lectura
    try:
        if uploaded.name.endswith(".csv"):
            try:
                df_raw = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, encoding="latin1")
        else:
            df_raw = pd.read_excel(uploaded, engine="openpyxl")
        st.success(f"✅ **{uploaded.name}** — {len(df_raw):,} filas · {df_raw.shape[1]} columnas")
    except Exception as e:
        st.error(f"❌ Error al leer: {e}")
        st.markdown("💡 Instala: `pip install fastexcel openpyxl`")
        st.stop()

    with st.expander("👁 Vista previa (5 filas)"):
        st.dataframe(df_raw.head(), use_container_width=True)

    st.divider()

    # ── MAPEO DE COLUMNAS ──
    st.subheader("🗂 Mapeo de Columnas")
    detected = autodetectar(df_raw.columns.tolist())
    col_opts = ["— No disponible —"] + df_raw.columns.tolist()

    c1, c2, c3 = st.columns(3)
    def picker(label, metric, col_widget, help=""):
        auto = detected.get(metric)
        idx  = col_opts.index(auto) if auto in col_opts else 0
        sel  = col_widget.selectbox(label, col_opts, index=idx, help=help, key=f"map_{metric}")
        return None if sel == "— No disponible —" else sel

    col_seg = picker("📂 Columna Segmento",  "segmento", c1, "Rama, plan, sector, categoría...")
    col_nps = picker("⭐ Columna Nota / NPS", "nps",      c2, "Puntaje numérico de satisfacción")
    col_txt = picker("💬 Columna Texto",      "texto",    c3, "Comentarios u opiniones abiertas")

    if not col_seg:
        st.warning("⚠️ Debes mapear al menos la columna **Segmento** para continuar.")
        st.stop()

    st.divider()

    # ── CONFIGURACIÓN DE ARQUETIPOS ──
    st.subheader("🏷 Arquetipos & Keywords")
    st.markdown("Revisa, edita o agrega arquetipos para tu tipo de cliente.")

    arquetipos_config = {}
    arq_base = perfil["arquetipos"].copy()

    # Si es Personalizado, el usuario agrega desde cero
    if tipo_cliente == "⚙️ Personalizado":
        n_arq = st.number_input("¿Cuántos arquetipos quieres definir?", 2, 10, 4)
        for i in range(int(n_arq)):
            ca, cb = st.columns([1, 3])
            nombre = ca.text_input(f"Nombre Arquetipo {i+1}", key=f"arq_name_{i}")
            kws    = cb.text_input(f"Keywords (separadas por coma)", key=f"arq_kw_{i}",
                                   placeholder="ej: rápido, bueno, recomiendo")
            if nombre:
                arquetipos_config[nombre] = [k.strip().lower() for k in kws.split(",") if k.strip()]
    else:
        for arq, kws_default in arq_base.items():
            ca, cb = st.columns([1, 3])
            nombre = ca.text_input("Arquetipo", value=arq, key=f"arq_n_{arq}")
            kws_str = cb.text_input(
                "Keywords",
                value=", ".join(kws_default),
                key=f"arq_k_{arq}",
                placeholder="palabra1, palabra2, ..."
            )
            if nombre:
                arquetipos_config[nombre] = [k.strip().lower() for k in kws_str.split(",") if k.strip()]

    # Guardar en session_state
    st.session_state["df_raw"]   = df_raw
    st.session_state["col_seg"]  = col_seg
    st.session_state["col_nps"]  = col_nps
    st.session_state["col_txt"]  = col_txt
    st.session_state["arquetipos"] = arquetipos_config

    st.success(f"✅ {len(arquetipos_config)} arquetipos configurados. Ve a la pestaña **② Muestra & Sintéticos**.")

# ════════════════════════════════════════════════════════
# TAB 2 — MUESTRA & SINTÉTICOS
# ════════════════════════════════════════════════════════
with tab2:

    if "df_raw" not in st.session_state:
        st.info("Primero completa la pestaña ① Datos & Configuración.")
        st.stop()

    df_raw      = st.session_state["df_raw"]
    col_seg     = st.session_state["col_seg"]
    col_nps     = st.session_state["col_nps"]
    col_txt     = st.session_state["col_txt"]
    arquetipos  = st.session_state["arquetipos"]

    st.subheader("📐 Parámetros Estadísticos")
    st.markdown("La fórmula de Cochran calcula el tamaño muestral mínimo para que tu muestra sea representativa.")

    p1, p2, p3, p4 = st.columns(4)

    N = p1.number_input(
        "N — Universo Total",
        min_value=100, max_value=10_000_000,
        value=5000, step=100,
        help="Total de personas en tu universo (socios, clientes, habitantes, etc.)"
    )
    confianza = p2.selectbox(
        "Z — Nivel de Confianza",
        options=["90% → Z=1.645", "95% → Z=1.960", "99% → Z=2.576"],
        index=1,
        help="Qué tan seguro quieres estar de que tu muestra representa al universo"
    )
    p_prop = p3.slider(
        "p — Proporción esperada",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="0.5 es el valor más conservador (máxima varianza)"
    )
    e_pct = p4.slider(
        "e — Margen de error",
        min_value=1, max_value=15,
        value=5, step=1,
        help="% de error aceptable en tus resultados"
    )

    seed = st.sidebar.number_input("🎲 Semilla aleatoria (reproducibilidad)", value=42, step=1)

    Z_map = {"90% → Z=1.645": 1.645, "95% → Z=1.960": 1.960, "99% → Z=2.576": 2.576}
    Z = Z_map[confianza]
    e = e_pct / 100

    n_necesaria = calcular_muestra(N, Z, p_prop, e)
    n_real      = len(df_raw)
    n_sinteticos = max(0, n_necesaria - n_real)
    cobertura   = min(100, round(n_real / n_necesaria * 100, 1))

    # ── RESUMEN METODOLÓGICO ──
    st.divider()
    st.subheader("📊 Resumen Metodológico")

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"""<div class="stat-box">
        <div class="stat-label">Universo Total</div>
        <div class="stat-value">{N:,}</div>
        <div class="stat-sub">personas / registros</div>
    </div>""", unsafe_allow_html=True)
    s2.markdown(f"""<div class="stat-box">
        <div class="stat-label">Muestra Necesaria</div>
        <div class="stat-value">{n_necesaria:,}</div>
        <div class="stat-sub">con {confianza.split()[0]} confianza, ±{e_pct}% error</div>
    </div>""", unsafe_allow_html=True)
    s3.markdown(f"""<div class="stat-box">
        <div class="stat-label">Encuestas Reales</div>
        <div class="stat-value">{n_real:,}</div>
        <div class="stat-sub">cobertura: {cobertura}%</div>
    </div>""", unsafe_allow_html=True)
    s4.markdown(f"""<div class="stat-box">
        <div class="stat-label">Sintéticos a Generar</div>
        <div class="stat-value" style="color:{'#4ecdc4' if n_sinteticos > 0 else '#aaa'}">{n_sinteticos:,}</div>
        <div class="stat-sub">{'generados por IA' if n_sinteticos > 0 else 'muestra suficiente ✅'}</div>
    </div>""", unsafe_allow_html=True)

    # Fórmula visible
    with st.expander("📐 Ver fórmula aplicada"):
        st.markdown(f"""<div class="method-box">
        Fórmula de Cochran (corrección población finita):<br><br>
        &nbsp;&nbsp;n₀ = (Z² × p × (1-p)) / e²<br>
        &nbsp;&nbsp;n₀ = ({Z}² × {p_prop} × {1-p_prop}) / {e}²  =  {math.ceil((Z**2 * p_prop * (1-p_prop)) / e**2):,}<br><br>
        &nbsp;&nbsp;n  = n₀ / (1 + (n₀ - 1) / N)<br>
        &nbsp;&nbsp;n  = {math.ceil((Z**2 * p_prop * (1-p_prop)) / e**2):,} / (1 + ({math.ceil((Z**2 * p_prop * (1-p_prop)) / e**2):,} - 1) / {N:,})<br><br>
        &nbsp;&nbsp;<b>n = {n_necesaria:,} encuestas mínimas</b>
        </div>""", unsafe_allow_html=True)

    if n_real >= n_necesaria:
        st.success(f"✅ Tu muestra real ({n_real:,}) ya supera la mínima necesaria ({n_necesaria:,}). No se requieren sintéticos.")

    st.divider()

    # ── EJECUTAR ──
    if st.button("🚀 Ejecutar Clasificación & Generar Universo", type="primary"):

        with st.spinner("Clasificando y generando datos sintéticos..."):

            # Clasificar reales
            df_proc = df_raw.copy()
            df_proc["Arquetipo"] = df_proc.apply(
                lambda row: clasificar_fila(
                    row[col_txt] if col_txt else "",
                    row[col_nps] if col_nps else 5,
                    arquetipos
                ), axis=1
            )
            df_proc["Origen"] = "Real (Encuesta)"

            # Generar sintéticos
            if n_sinteticos > 0:
                df_sint = generar_sinteticos(
                    df_proc[[col_seg, "Arquetipo", "Origen"]],
                    col_seg, "Arquetipo", n_sinteticos, seed=int(seed)
                )
                df_final = pd.concat([df_proc, df_sint], ignore_index=True)
            else:
                df_final = df_proc.copy()

        st.session_state["df_final"] = df_final
        st.session_state["col_seg_final"] = col_seg

        st.success(f"✅ Universo generado: {len(df_final):,} registros ({n_real:,} reales + {n_sinteticos:,} sintéticos)")

        # ── VISUALIZACIONES ──
        tab_a, tab_b, tab_c = st.tabs(["Distribución Arquetipos", "Por Segmento", "Real vs Sintético"])

        with tab_a:
            dist = df_final["Arquetipo"].value_counts().reset_index()
            dist.columns = ["Arquetipo", "Cantidad"]
            fig = px.bar(dist, x="Arquetipo", y="Cantidad",
                         color="Arquetipo", title="Distribución de Arquetipos en el Universo")
            st.plotly_chart(fig, use_container_width=True)

        with tab_b:
            matriz = pd.crosstab(df_final[col_seg], df_final["Arquetipo"], normalize="index")
            fig2 = px.imshow(
                matriz.round(2),
                text_auto=".0%",
                color_continuous_scale="Blues",
                title="Mapa Estratégico: Segmento × Arquetipo"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab_c:
            origen_dist = df_final["Origen"].value_counts().reset_index()
            origen_dist.columns = ["Origen", "Cantidad"]
            fig3 = px.pie(origen_dist, names="Origen", values="Cantidad",
                          title="Composición Real vs Sintético",
                          color_discrete_map={"Real (Encuesta)": "#4ecdc4", "Sintético (IA)": "#2a5f7f"})
            st.plotly_chart(fig3, use_container_width=True)

        # ── DESCARGA ──
        st.divider()
        csv_out = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar Universo Completo (CSV)",
            csv_out, "universo_ia.csv", "text/csv"
        )

        st.markdown("➡️ Ve a la pestaña **③ Clusterización** para segmentar el universo generado.")

# ════════════════════════════════════════════════════════
# TAB 3 — CLUSTERIZACIÓN
# ════════════════════════════════════════════════════════
with tab3:

    if "df_final" not in st.session_state:
        st.info("Primero ejecuta el análisis en la pestaña ② Muestra & Sintéticos.")
        st.stop()

    df_final  = st.session_state["df_final"]
    col_seg_f = st.session_state.get("col_seg_final", "")

    st.subheader("🔬 Clusterización K-Means")
    st.markdown("Agrupa registros basándose en variables numéricas para encontrar patrones no obvios.")

    # Columnas numéricas disponibles
    cols_num = df_final.select_dtypes(include=[np.number]).columns.tolist()

    if len(cols_num) < 2:
        st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para clusterizar. Tu dataset tiene pocas variables numéricas.")
        st.info("💡 Tip: asegúrate de que columnas como NPS, edad, frecuencia, ticket sean numéricas en tu Excel.")
        st.stop()

    cols_cluster = st.multiselect(
        "Variables para clusterizar",
        cols_num,
        default=cols_num[:min(4, len(cols_num))],
        help="Selecciona las variables numéricas más relevantes para segmentar"
    )

    if len(cols_cluster) < 2:
        st.warning("Selecciona al menos 2 variables.")
        st.stop()

    modo_k = st.radio("¿Cómo definir el número de clusters (K)?",
                       ["🤖 Automático (IA elige el K óptimo)", "✋ Manual"],
                       horizontal=True)

    if modo_k == "✋ Manual":
        k_elegido = st.slider("Número de clusters", 2, 10, 4)
    else:
        k_elegido = None

    if st.button("🔬 Ejecutar Clusterización", type="primary"):

        with st.spinner("Buscando K óptimo y clusterizando..."):

            df_cluster_input = df_final[cols_cluster].fillna(df_final[cols_cluster].median())
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster_input)

            if k_elegido is None:
                k_opt, ks, inertias, silhouettes = encontrar_k_optimo(X_scaled)
                st.info(f"🤖 K óptimo detectado: **{k_opt} clusters** (mayor Silhouette Score)")

                # Gráfico del codo
                fig_codo = make_subplots(rows=1, cols=2,
                    subplot_titles=["Método del Codo (Inercia)", "Silhouette Score"])
                fig_codo.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                    name="Inercia", line=dict(color="#4ecdc4")), row=1, col=1)
                fig_codo.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers",
                    name="Silhouette", line=dict(color="#ff6b6b")), row=1, col=2)
                fig_codo.update_layout(title="Selección automática de K óptimo", showlegend=False)
                st.plotly_chart(fig_codo, use_container_width=True)
                k_final = k_opt
            else:
                k_final = k_elegido

            df_resultado, _ = clusterizar(df_final, cols_cluster, k_final)

        st.session_state["df_resultado"] = df_resultado

        st.success(f"✅ {k_final} clusters identificados sobre {len(df_resultado):,} registros.")

        # ── VISUALIZACIONES CLUSTER ──
        ct1, ct2, ct3 = st.tabs(["Distribución Clusters", "Clusters × Segmento", "Perfil Numérico"])

        with ct1:
            dist_c = df_resultado["Cluster"].value_counts().reset_index()
            dist_c.columns = ["Cluster", "Cantidad"]
            dist_c["Cluster"] = "Cluster " + dist_c["Cluster"]
            fig_c1 = px.bar(dist_c, x="Cluster", y="Cantidad",
                            color="Cluster", title="Tamaño de cada Cluster")
            st.plotly_chart(fig_c1, use_container_width=True)

        with ct2:
            if col_seg_f and col_seg_f in df_resultado.columns:
                mat_c = pd.crosstab(df_resultado[col_seg_f], df_resultado["Cluster"])
                fig_c2 = px.imshow(mat_c, text_auto=True,
                                   color_continuous_scale="Blues",
                                   title="Clusters por Segmento")
                st.plotly_chart(fig_c2, use_container_width=True)
            else:
                st.info("Mapea la columna Segmento en la pestaña ① para ver este gráfico.")

        with ct3:
            perfil_num = df_resultado.groupby("Cluster")[cols_cluster].mean().round(2)
            st.dataframe(perfil_num, use_container_width=True)
            fig_c3 = px.imshow(perfil_num,
                                text_auto=".1f",
                                color_continuous_scale="RdBu_r",
                                title="Perfil Numérico Promedio por Cluster")
            st.plotly_chart(fig_c3, use_container_width=True)

        # ── DESCARGA FINAL ──
        st.divider()
        csv_final = df_resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar Dataset Completo con Clusters (CSV)",
            csv_final, "universo_clusterizado.csv", "text/csv"
        )
