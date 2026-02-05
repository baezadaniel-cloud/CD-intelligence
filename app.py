import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Motor de Segmentación IA", layout="wide", page_icon="🧬")

st.title("🧬 Motor de Segmentación de Comunidades con IA")
st.markdown("""
**Sistema de Gemelos Digitales & Data Augmentation.**
Sube tus datos reales y la IA generará el universo completo proyectado.
""")

# --- 2. FUNCIONES DE CARGA ---
def cargar_big_data(file):
    """Carga robusta con feedback de error."""
    try:
        filename = file.name.lower()
        if filename.endswith('.parquet'):
            return pl.read_parquet(file).to_pandas()
        elif filename.endswith('.csv'):
            try:
                return pl.read_csv(file, ignore_errors=True).to_pandas()
            except:
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None, str(e) # Retorna el error específico
    return None, "Formato no soportado"

# --- 3. LÓGICA DEL MODELO ---
class CommunityAI:
    def __init__(self):
        self.knowledge_base = None 
        self.real_sample_size = 0
        self.global_probs = None
        
    def entrenar(self, df, col_segmento, col_nps, cols_texto):
        self.real_sample_size = len(df)
        df['NLP_Context'] = df[cols_texto].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        def detectar_arquetipo(row):
            txt = row['NLP_Context']
            score = pd.to_numeric(row[col_nps], errors='coerce')
            if any(x in txt for x in ['luz', 'baño', 'agua', 'sucio', 'infra', 'cancha']): return 'Crítico (Infraestructura)'
            if any(x in txt for x in ['ganar', 'compet', 'torneo', 'ranking', 'copa']): return 'Competitivo (Logro)'
            if any(x in txt for x in ['amigo', 'social', 'grupo', 'ambiente', 'asado']): return 'Social (Pertenencia)'
            if any(x in txt for x in ['clase', 'profe', 'aprender', 'escuela']): return 'Formativo (Desarrollo)'
            if score >= 6: return 'Promotor Silencioso'
            if score <= 4: return 'Detractor Silencioso'
            return 'Neutro / Pasivo'

        df['Arquetipo'] = df.apply(detectar_arquetipo, axis=1)
        self.knowledge_base = pd.crosstab(df[col_segmento], df['Arquetipo'], normalize='index')
        self.global_probs = df['Arquetipo'].value_counts(normalize=True)
        return df

    def generar_sinteticos(self, df_real, col_segmento, total_universo):
        if self.knowledge_base is None: return df_real, 0
        
        faltantes = total_universo - self.real_sample_size
        if faltantes <= 0: return df_real, 0
        
        distribucion_segmentos = df_real[col_segmento].value_counts(normalize=True)
        nuevos_segmentos = np.random.choice(distribucion_segmentos.index, size=faltantes, p=distribucion_segmentos.values)
        
        sinteticos = []
        for seg in nuevos_segmentos:
            probs = self.knowledge_base.loc[seg] if seg in self.knowledge_base.index else self.global_probs
            perfil = np.random.choice(probs.index, p=probs.values)
            sinteticos.append({col_segmento: seg, 'Arquetipo': perfil, 'Origen_Dato': 'Sintético (IA)'})
            
        df_sintetico = pd.DataFrame(sinteticos)
        df_real_copy = df_real[[col_segmento, 'Arquetipo']].copy()
        df_real_copy['Origen_Dato'] = 'Real (Encuesta)'
        
        return pd.concat([df_real_copy, df_sintetico], ignore_index=True), faltantes

# --- 4. INTERFAZ MEJORADA ---

st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo (Excel/CSV)", type=['csv','xlsx','parquet'])
total_universo = st.sidebar.number_input("Universo Total a Proyectar", min_value=100, value=5000, step=100)

if uploaded_file:
    # Intento de Carga
    res = cargar_big_data(uploaded_file)
    
    # Verificación de Errores
    if isinstance(res, tuple): # Si devolvió error
        st.error(f"❌ Error al leer el archivo: {res[1]}")
        st.info("Prueba guardando tu archivo como CSV UTF-8 o Excel estándar (.xlsx)")
    
    elif res is not None:
        df_raw = res
        st.success(f"✅ Archivo cargado correctamente: {len(df_raw)} filas detectadas.")
        
        # --- VISTA PREVIA (Para que el usuario sepa que funcionó) ---
        with st.expander("🔍 Ver Vista Previa de los Datos", expanded=True):
            st.dataframe(df_raw.head(5))

        st.divider()
        st.subheader("2. Configuración de Variables")
        st.info("Selecciona las columnas correspondientes para entrenar a la IA.")
        
        c1, c2, c3 = st.columns(3)
        col_seg = c1.selectbox("Columna Segmento (Ej: Rama/Área)", df_raw.columns)
        col_nps = c2.selectbox("Columna Numérica (Ej: NPS/Nota)", df_raw.columns)
        cols_txt = c3.multiselect("Columnas de Texto (Comentarios)", df_raw.columns)
        
        st.divider()
        
        # --- BOTÓN DE ACCIÓN (GRANDE) ---
        if st.button("🚀 INICIAR SIMULACIÓN DE UNIVERSO", type="primary", use_container_width=True):
            if not cols_txt:
                st.error("⚠️ Por favor selecciona al menos una columna de texto.")
            else:
                engine = CommunityAI()
                with st.spinner("🧠 Entrenando modelo y generando gemelos digitales..."):
                    df_etiquetado = engine.entrenar(df_raw, col_seg, col_nps, cols_txt)
                    df_full, n_gen = engine.generar_sinteticos(df_etiquetado, col_seg, total_universo)
                
                # --- RESULTADOS ---
                st.balloons()
                st.success("¡Análisis Completado!")
                
                # Métricas
                m1, m2, m3 = st.columns(3)
                m1.metric("Muestra Real", f"{len(df_raw)}")
                m2.metric("Sintéticos Generados", f"{n_gen}")
                m3.metric("Universo Final", f"{len(df_full)}")
                
                # Gráfico
                st.subheader("Mapa de Calor Estratégico")
                heatmap_data = pd.crosstab(df_full[col_seg], df_full['Arquetipo'], normalize='index') * 100
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Blues", ax=ax)
                st.pyplot(fig)
                
                # Descarga
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_full.to_excel(writer, sheet_name='Data_Completa', index=False)
                st.download_button("📥 Descargar Reporte Final (.xlsx)", buffer, "Reporte_IA.xlsx")

else:
    st.info("👈 Sube un archivo en el menú lateral para comenzar.")
