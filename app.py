import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- IMPORTACIÓN ROBUSTA (Evita pantalla blanca si falta Polars) ---
try:
    import polars as pl
except ImportError:
    st.error("❌ ERROR CRÍTICO: Falta instalar 'polars'. Revisa requirements.txt")
    st.stop()

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Motor IA", layout="wide")

st.title("🧬 Motor de Segmentación de Comunidades con IA")
st.write("---") # Línea divisoria

# --- 2. FUNCIONES DE CARGA (Con avisos) ---
def cargar_datos(file):
    status = st.empty() # Espacio para mensajes temporales
    status.info("⏳ Leyendo archivo...")
    
    try:
        filename = file.name.lower()
        df = None
        
        if filename.endswith('.parquet'):
            df = pl.read_parquet(file).to_pandas()
        elif filename.endswith('.csv'):
            try:
                df = pl.read_csv(file, ignore_errors=True).to_pandas()
            except:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin1')
        else:
            # Excel
            df = pd.read_excel(file)
            
        status.success("✅ Archivo leído correctamente en memoria.")
        return df
        
    except Exception as e:
        status.error(f"❌ Error leyendo el archivo: {str(e)}")
        return None

# --- 3. CEREBRO (Simplificado para debug) ---
class CommunityAI:
    def entrenar_y_generar(self, df, col_seg, col_nps, cols_txt, total):
        # 1. Etiquetado simple
        df['Texto'] = df[cols_txt].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        def perfil(row):
            t = row['Texto']
            s = pd.to_numeric(row[col_nps], errors='coerce')
            if 'infra' in t or 'luz' in t or 'baño' in t: return 'Crítico Infraestructura'
            if 'ganar' in t or 'compet' in t: return 'Competitivo'
            if 'social' in t or 'amigos' in t: return 'Social'
            if s >= 6: return 'Satisfecho'
            return 'Neutro'
            
        df['Arquetipo'] = df.apply(perfil, axis=1)
        
        # 2. Generación Sintética
        real_count = len(df)
        faltan = total - real_count
        
        if faltan <= 0: return df
        
        # Probabilidades simples
        probs = df['Arquetipo'].value_counts(normalize=True)
        seg_probs = df[col_seg].value_counts(normalize=True)
        
        # Generar
        nuevos_seg = np.random.choice(seg_probs.index, size=faltan, p=seg_probs.values)
        nuevos_arq = np.random.choice(probs.index, size=faltan, p=probs.values)
        
        df_new = pd.DataFrame({
            col_seg: nuevos_seg,
            'Arquetipo': nuevos_arq,
            'Origen': 'Sintético'
        })
        
        df['Origen'] = 'Real'
        return pd.concat([df[[col_seg, 'Arquetipo', 'Origen']], df_new])

# --- 4. INTERFAZ CENTRAL (Sin Sidebar) ---
st.subheader("1. Carga de Datos")
c1, c2 = st.columns([2, 1])

# SUBIDA DE ARCHIVO (En el centro, no en sidebar)
uploaded_file = c1.file_uploader("Arrastra tu Excel o CSV aquí", type=['xlsx','csv','parquet'])
total_universo = c2.number_input("Universo Total a Proyectar", value=5000)

if uploaded_file is not None:
    # Paso 1: Carga
    df = cargar_datos(uploaded_file)
    
    if df is not None:
        # Paso 2: Mostrar data cruda (Prueba de vida)
        st.write(f"**Vista previa ({len(df)} registros reales):**")
        st.dataframe(df.head(3))
        
        # Paso 3: Selectores
        st.subheader("2. Configuración")
        col_seg = st.selectbox("Columna Segmento (Ej: Rama)", df.columns)
        col_nps = st.selectbox("Columna Nota/NPS", df.columns)
        cols_txt = st.multiselect("Columnas de Texto", df.columns)
        
        # Paso 4: Botón
        st.write("---")
        if st.button("🚀 EJECUTAR ANÁLISIS AHORA", type="primary"):
            if not cols_txt:
                st.error("Selecciona columnas de texto.")
            else:
                st.info("🧠 Procesando... Por favor espera.")
                ai = CommunityAI()
                
                try:
                    df_final = ai.entrenar_y_generar(df, col_seg, col_nps, cols_txt, total_universo)
                    
                    st.success("¡Éxito! Resultados generados.")
                    
                    # Gráfico
                    c_chart, c_metrics = st.columns([2,1])
                    
                    with c_chart:
                        st.write("**Mapa de Calor:**")
                        matriz = pd.crosstab(df_final[col_seg], df_final['Arquetipo'], normalize='index')
                        fig, ax = plt.subplots()
                        sns.heatmap(matriz, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                        st.pyplot(fig)
                        
                    with c_metrics:
                        st.metric("Total Universo", len(df_final))
                        st.metric("Datos Reales", len(df))
                        st.metric("Datos Sintéticos", len(df_final)-len(df))
                        
                    # Descarga
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_final.to_excel(writer, index=False)
                    
                    st.download_button("📥 Descargar Excel", buffer, "resultado_ia.xlsx")
                    
                except Exception as e:
                    st.error(f"Error en el cálculo: {e}")

else:
    st.info("👆 Esperando archivo...")
