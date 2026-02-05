import streamlit as st
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Universal Community Analyzer", layout="wide", page_icon="🌐")

st.title("🌐 Motor de Segmentación de Comunidades con IA")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para cruzar datos demográficos con datos psicográficos (opiniones).
**Aplicable a:** Clubes Deportivos, RRHH (Clima Laboral), Educación, Retail y Consorcios.
""")

# --- FUNCIONES ---
def carga_inteligente(file):
    try:
        if file.name.endswith('.parquet'): return pl.read_parquet(file).to_pandas()
        elif file.name.endswith('.csv'):
            try: return pl.read_csv(file, ignore_errors=True).to_pandas()
            except: 
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else: return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error de lectura: {e}")
        return None

# --- CLASE CORE (MOTOR DE IA) ---
class CommunityDigitalTwin:
    def __init__(self):
        self.profiles_prob = None
        self.global_prob = None

    def fit(self, df, col_grupo, col_score, cols_text):
        # 1. NLP Básico
        df['Texto_Full'] = df[cols_text].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        # 2. Motor de Clasificación Universal
        def assign_profile(row):
            t = row['Texto_Full']
            n = pd.to_numeric(row[col_score], errors='coerce')
            
            # Palabras Clave Universales
            if 'luz' in t or 'baño' in t or 'sucio' in t or 'infra' in t or 'lento' in t: return 'Crítico (Infraestructura)'
            if 'ganar' in t or 'compet' in t or 'meta' in t or 'bono' in t or 'precio' in t: return 'Orientado a Logro/Precio'
            if 'social' in t or 'amigos' in t: return 'Social/Comunidad'
            if 'aprender' in t or 'curso' in t or 'profe' in t: return 'Formativo/Crecimiento'
            
            # Score Numérico
            if n >= 6: return 'Promotor (Satisfecho)'
            if n <= 4: return 'Detractor (Riesgo)'
            return 'Neutro (Pasivo)'

        df['Perfil'] = df.apply(assign_profile, axis=1)
        
        # 3. Matriz Probabilística
        self.profiles_prob = pd.crosstab(df[col_grupo], df['Perfil'], normalize='index')
        self.global_prob = df['Perfil'].value_counts(normalize=True)
        return True

    def generate(self, df_univ, col_group, col_qty):
        if self.profiles_prob is None: return None
        pop = []
        for _, row in df_univ.iterrows():
            try:
                group = str(row[col_group])
                qty = int(float(str(row[col_qty]).replace('.','').replace(',','')))
                if qty > 0:
                    match = [i for i in self.profiles_prob.index if str(i).lower() in group.lower()]
                    dist = self.profiles_prob.loc[match[0]] if match else self.global_prob
                    clones = np.random.choice(dist.index, size=qty, p=dist.values)
                    for c in clones: pop.append({'Group': group, 'Profile': c})
            except: continue
        return pd.DataFrame(pop)

# --- INTERFAZ ---
st.info("💡 Sube tus archivos para comenzar. El sistema detectará patrones automáticamente.")

c1, c2 = st.columns(2)
f_encuesta = c1.file_uploader("1. Datos Cualitativos (Encuesta/Opiniones)", type=['csv','xlsx'])
f_censo = c2.file_uploader("2. Datos Cuantitativos (Censo/Total)", type=['csv','xlsx'])

if f_encuesta and f_censo:
    df_e = carga_inteligente(f_encuesta)
    df_c = carga_inteligente(f_censo)
    
    if df_e is not None and df_c is not None:
        st.divider()
        with st.expander("⚙️ Configurar Variables (Click para abrir)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Datos de Opinión")
                col_rama = st.selectbox("Variable de Agrupación (ej: Rama, Depto)", df_e.columns)
                col_nps = st.selectbox("Variable Numérica (ej: NPS, Nota)", df_e.columns)
                cols_text = st.multiselect("Variables de Texto (Comentarios)", df_e.columns)
            with col2:
                st.subheader("Datos de Universo")
                col_group = st.selectbox("Variable de Grupo (ej: Rama, Depto)", df_c.columns)
                col_qty = st.selectbox("Variable de Cantidad Total", df_c.columns)

        if st.button("🚀 Analizar Universo Completo", type="primary"):
            bot = CommunityDigitalTwin()
            
            if cols_text and col_rama:
                bot.fit(df_e, col_rama, col_nps, cols_text)
                df_final = bot.generate(df_c, col_group, col_qty)
                
                if df_final is not None:
                    st.divider()
                    st.success("Análisis Completado Exitosamente")
                    
                    # 1. GRÁFICO
                    st.subheader("📍 Mapa de Calor Estratégico")
                    st.caption("Eje X: Orientación Social/Comunidad | Eje Y: Orientación Logro/Rendimiento")
                    
                    res = pd.crosstab(df_final['Group'], df_final['Profile'], normalize='index') * 100
                    res['Size'] = df_final['Group'].value_counts()
                    
                    # Garantizar ejes para el plot
                    if 'Social/Comunidad' not in res.columns: res['Social/Comunidad'] = 0
                    if 'Orientado a Logro/Precio' not in res.columns: res['Orientado a Logro/Precio'] = 0
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=res, x='Social/Comunidad', y='Orientado a Logro/Precio', size='Size', sizes=(100, 1000), hue=res.index, alpha=0.7, ax=ax, legend=False)
                    
                    for i in range(len(res)):
                        ax.text(res['Social/Comunidad'].iloc[i], res['Orientado a Logro/Precio'].iloc[i]+1, res.index[i], ha='center', size=8)
                        
                    st.pyplot(fig)
                    
                    # 2. DESCARGA
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_final.to_excel(writer, sheet_name='Data_Sintetica', index=False)
                    
                    st.download_button("📥 Descargar Reporte (.xlsx)", buffer, "reporte_universal.xlsx")
            else:
                st.warning("Por favor selecciona las columnas de texto.")
