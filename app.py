import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- 1. CONFIGURACIÓN VISUAL PRO ---
st.set_page_config(page_title="Motor de Segmentación IA", layout="wide", page_icon="🧬")

st.title("🧬 Motor de Segmentación de Comunidades con IA")
st.markdown("""
**Sistema de Gemelos Digitales & Data Augmentation.**
Ingesta datos muestrales (Encuestas/Big Data), detecta patrones psicográficos y genera poblaciones sintéticas para completar el universo total.
""")

# --- 2. MOTOR DE CARGA (POLARS ENGINE) ---
def cargar_big_data(file):
    """
    Detecta formato y usa Polars para máxima velocidad.
    Retorna Pandas DataFrame para compatibilidad con gráficos.
    """
    try:
        filename = file.name.lower()
        if filename.endswith('.parquet'):
            # Polars nativo (Velocidad extrema)
            return pl.read_parquet(file).to_pandas()
        elif filename.endswith('.csv'):
            try:
                # Intentamos Polars primero
                return pl.read_csv(file, ignore_errors=True).to_pandas()
            except:
                # Fallback a Pandas si el CSV está "sucio"
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else:
            # Excel (Lento pero seguro)
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error crítico en lectura de datos: {e}")
        return None

# --- 3. CEREBRO DEL MODELO (LÓGICA HÍBRIDA) ---
class CommunityAI:
    def __init__(self):
        self.knowledge_base = None # Aquí guardamos las probabilidades aprendidas
        self.real_sample_size = 0
        
    def entrenar(self, df, col_segmento, col_nps, cols_texto):
        """
        Fase 1: Entendimiento (Learning Phase).
        Aprende qué mueve a cada segmento basándose en la muestra real.
        """
        self.real_sample_size = len(df)
        
        # A. Procesamiento de Lenguaje Natural (NLP Básico)
        # Unimos todas las columnas de texto en una "sopa de palabras"
        df['NLP_Context'] = df[cols_texto].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        # B. Reglas de Inferencia Psicográfica (Etiquetado)
        def detectar_arquetipo(row):
            txt = row['NLP_Context']
            score = pd.to_numeric(row[col_nps], errors='coerce')
            
            # 1. Detectar Dolores (Pain Points)
            if any(x in txt for x in ['luz', 'baño', 'agua', 'sucio', 'infra', 'cancha', 'estacionamiento']):
                return 'Crítico (Infraestructura)'
            
            # 2. Detectar Motivaciones de Logro
            if any(x in txt for x in ['ganar', 'compet', 'torneo', 'ranking', 'copa', 'medalla', 'nivel']):
                return 'Competitivo (Logro)'
            
            # 3. Detectar Motivaciones Sociales
            if any(x in txt for x in ['amigo', 'social', 'grupo', 'ambiente', 'asado', 'tercer', 'familia']):
                return 'Social (Pertenencia)'
            
            # 4. Detectar Motivaciones de Crecimiento
            if any(x in txt for x in ['clase', 'profe', 'aprender', 'escuela', 'mejorar', 'tecnica']):
                return 'Formativo (Desarrollo)'
            
            # 5. Si no hay texto claro, usamos el NPS (Behavioral)
            if score >= 6: return 'Promotor Silencioso'
            if score <= 4: return 'Detractor Silencioso'
            
            return 'Neutro / Pasivo'

        df['Arquetipo'] = df.apply(detectar_arquetipo, axis=1)
        
        # C. Generar Matriz de Probabilidades (El "Cerebro")
        # Calculamos % de cada arquetipo POR segmento (ej: % de Competitivos en Tenis vs Futbol)
        self.knowledge_base = pd.crosstab(df[col_segmento], df['Arquetipo'], normalize='index')
        self.global_probs = df['Arquetipo'].value_counts(normalize=True)
        
        return df # Retornamos la data real ya etiquetada

    def generar_sinteticos(self, df_real, col_segmento, total_universo):
        """
        Fase 2: Generación (Data Augmentation).
        Usa Simulación de Montecarlo para completar el universo.
        """
        if self.knowledge_base is None: return None
        
        faltantes = total_universo - self.real_sample_size
        
        if faltantes <= 0:
            return df_real # No necesitamos generar nada
        
        # Distribución de segmentos en la muestra real
        distribucion_segmentos = df_real[col_segmento].value_counts(normalize=True)
        
        sinteticos = []
        
        # Generamos los agentes faltantes
        # Paso 1: ¿A qué segmento pertenecen los nuevos? (Asumimos misma distribución que la real)
        nuevos_segmentos = np.random.choice(distribucion_segmentos.index, 
                                            size=faltantes, 
                                            p=distribucion_segmentos.values)
        
        # Paso 2: ¿Qué psicología tienen? (Basado en la Matriz de Probabilidad de su segmento)
        for seg in nuevos_segmentos:
            # Buscamos la "personalidad" de ese segmento
            if seg in self.knowledge_base.index:
                probs = self.knowledge_base.loc[seg]
                perfil = np.random.choice(probs.index, p=probs.values)
            else:
                # Si es un segmento nuevo desconocido, usamos el promedio global
                perfil = np.random.choice(self.global_probs.index, p=self.global_probs.values)
            
            sinteticos.append({
                col_segmento: seg,
                'Arquetipo': perfil,
                'Origen_Dato': 'Sintético (IA)'
            })
            
        df_sintetico = pd.DataFrame(sinteticos)
        
        # Marcamos la data real
        df_real_copy = df_real[[col_segmento, 'Arquetipo']].copy()
        df_real_copy['Origen_Dato'] = 'Real (Encuesta)'
        
        # Unimos Real + Sintético
        df_final = pd.concat([df_real_copy, df_sintetico], ignore_index=True)
        
        return df_final, faltantes

# --- 4. INTERFAZ DE USUARIO ---

# Panel Lateral de Control
st.sidebar.header("📁 Panel de Control")
uploaded_file = st.sidebar.file_uploader("Subir Muestra (Real)", type=['csv','xlsx','parquet'])
total_universo = st.sidebar.number_input("Tamaño Total del Universo (Socios Totales)", min_value=100, value=5000, step=100)

if uploaded_file:
    # Carga
    df_raw = cargar_big_data(uploaded_file)
    
    if df_raw is not None:
        st.sidebar.success(f"Cargados {len(df_raw)} registros reales.")
        
        # Configuración de Columnas
        st.subheader("⚙️ Configuración del Motor de IA")
        c1, c2, c3 = st.columns(3)
        col_seg = c1.selectbox("Variable de Segmentación (Rama/Área)", df_raw.columns)
        col_nps = c2.selectbox("Variable Cuantitativa (NPS/Nota)", df_raw.columns)
        cols_txt = c3.multiselect("Variables Cualitativas (Texto)", df_raw.columns)
        
        if st.button("🚀 Ejecutar Simulación de Universo", type="primary"):
            if not cols_txt:
                st.error("Debes seleccionar al menos una columna de texto para el análisis psicográfico.")
            else:
                engine = CommunityAI()
                
                with st.spinner("1. Analizando patrones psicográficos en data real..."):
                    df_etiquetado = engine.entrenar(df_raw, col_seg, col_nps, cols_txt)
                
                with st.spinner(f"2. Generando agentes sintéticos para llegar a {total_universo}..."):
                    df_full, n_gen = engine.generar_sinteticos(df_etiquetado, col_seg, total_universo)
                
                # --- RESULTADOS ---
                st.divider()
                st.markdown(f"### 📊 Resultados del Universo Proyectado (N={len(df_full)})")
                
                # Métricas Clave
                m1, m2, m3 = st.columns(3)
                m1.metric("Muestra Real Analizada", f"{len(df_raw)}")
                m2.metric("Data Sintética Generada", f"{n_gen}", delta="Data Augmentation")
                m3.metric("Cobertura Total", "100%")
                
                # Gráfico Estratégico
                st.subheader("Mapa de Calor de Comunidades")
                
                # Preparamos data para heatmap
                heatmap_data = pd.crosstab(df_full[col_seg], df_full['Arquetipo'], normalize='index') * 100
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': '% Afinidad'}, ax=ax)
                ax.set_ylabel("Segmento / Rama")
                ax.set_xlabel("Arquetipo Psicográfico")
                st.pyplot(fig)
                
                # Descarga
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_full.to_excel(writer, sheet_name='Universo_Completo', index=False)
                    heatmap_data.to_excel(writer, sheet_name='Matriz_Estrategica')
                
                st.download_button(
                    label="📥 Descargar Universo Completo (.xlsx)",
                    data=buffer,
                    file_name="Universo_Sintetico_IA.xlsx",
                    mime="application/vnd.ms-excel"
                )

else:
    st.info("👈 Sube un archivo en el panel lateral para comenzar.")
    st.markdown("### ¿Cómo funciona?")
    st.markdown("""
    1. **Ingesta:** Subes los datos que tienes (ej: 1,680 respuestas).
    2. **Aprendizaje:** La IA detecta qué motiva a cada grupo (Reglas + Probabilidad).
    3. **Expansión:** Si tu comunidad real son 5,000 personas, la IA genera los 3,320 perfiles faltantes basándose en los patrones estadísticos detectados.
    """)  
