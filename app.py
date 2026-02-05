import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Motor IA", layout="wide")
st.title("🧬 Motor de Segmentación de Comunidades con IA")
st.markdown("**Sistema de Gemelos Digitales & Data Augmentation.**")

# --- 2. CARGA DE DATOS (Modo Seguro Pandas) ---
def cargar_datos_seguro(file):
    try:
        if file.name.endswith('.csv'):
            try:
                return pd.read_csv(file, encoding='utf-8')
            except:
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else:
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error de lectura: {e}")
        return None

# --- 3. CEREBRO IA (Lógica mantenida) ---
class CommunityAI:
    def procesar(self, df, col_seg, col_nps, cols_txt, total_universo):
        # A. Etiquetado
        df['Texto_Full'] = df[cols_txt].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        def clasificar(row):
            txt = row['Texto_Full']
            try:
                score = float(row[col_nps])
            except:
                score = 0
            
            if any(x in txt for x in ['infra', 'luz', 'baño', 'cancha', 'agua', 'sucio']): return 'Crítico Infraestructura'
            if any(x in txt for x in ['compet', 'ganar', 'torneo', 'copa', 'medal']): return 'Competitivo'
            if any(x in txt for x in ['social', 'amigo', 'grupo', 'asado', 'fies']): return 'Social'
            if any(x in txt for x in ['clase', 'profe', 'aprender', 'taller']): return 'Formativo'
            if score >= 6: return 'Promotor'
            if score <= 4: return 'Detractor'
            return 'Neutro'

        df['Arquetipo'] = df.apply(clasificar, axis=1)
        
        # B. Generación de Sintéticos
        real_count = len(df)
        faltantes = total_universo - real_count
        
        if faltantes <= 0:
            df['Origen'] = 'Real'
            return df
            
        # Cálculo de probabilidades
        dist_seg = df[col_seg].value_counts(normalize=True)
        # Probabilidad de Arquetipo DADO el Segmento
        matrix_prob = pd.crosstab(df[col_seg], df['Arquetipo'], normalize='index')
        global_prob = df['Arquetipo'].value_counts(normalize=True)
        
        # Generamos
        nuevos_seg = np.random.choice(dist_seg.index, size=faltantes, p=dist_seg.values)
        sinteticos = []
        
        for seg in nuevos_seg:
            if seg in matrix_prob.index:
                probs = matrix_prob.loc[seg]
                arq = np.random.choice(probs.index, p=probs.values)
            else:
                arq = np.random.choice(global_prob.index, p=global_prob.values)
            sinteticos.append({col_seg: seg, 'Arquetipo': arq, 'Origen': 'Sintético (IA)'})
            
        df_sintetico = pd.DataFrame(sinteticos)
        df_real = df[[col_seg, 'Arquetipo']].copy()
        df_real['Origen'] = 'Real (Encuesta)'
        
        return pd.concat([df_real, df_sintetico], ignore_index=True)

# --- 4. INTERFAZ ---
uploaded_file = st.file_uploader("📂 Sube tu Excel o CSV aquí", type=['xlsx','csv'])
total_universo = st.number_input("Universo Total a Proyectar", value=5000, step=100)

if uploaded_file:
    df = cargar_datos_seguro(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Datos cargados: {len(df)} registros.")
        with st.expander("Ver datos cargados"):
            st.dataframe(df.head())
            
        st.divider()
        c1, c2, c3 = st.columns(3)
        col_seg = c1.selectbox("Columna Segmento (Rama)", df.columns)
        col_nps = c2.selectbox("Columna Nota/NPS", df.columns)
        cols_txt = c3.multiselect("Columnas Texto", df.columns)
        
        if st.button("🚀 EJECUTAR ANÁLISIS", type="primary"):
            if cols_txt:
                ai = CommunityAI()
                with st.spinner("Generando universo..."):
                    df_final = ai.procesar(df, col_seg, col_nps, cols_txt, total_universo)
                
                st.balloons()
                
                # Métricas
                m1, m2 = st.columns(2)
                m1.metric("Muestra Real", len(df))
                m2.metric("Universo Proyectado", len(df_final))
                
                # Gráfico
                st.subheader("Mapa Estratégico")
                matriz = pd.crosstab(df_final[col_seg], df_final['Arquetipo'], normalize='index')
                fig, ax = plt.subplots(figsize=(10,5))
                sns.heatmap(matriz, annot=True, fmt=".1%", cmap="Blues", ax=ax)
                st.pyplot(fig)
                
                # Descarga (Método estándar CSV para máxima compatibilidad)
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Resultado (CSV)", csv, "universo_ia.csv", "text/csv")
            else:
                st.error("Selecciona columnas de texto.")
