import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="CDUC Intelligence", layout="wide")

st.title("📊 CDUC: Plataforma de Inteligencia de Comunidades")
st.markdown("""
Esta herramienta automatiza el análisis de segmentación psicográfica.
Suba los archivos de **Encuesta** y **Censo** para generar el diagnóstico estratégico.
""")

# --- FUNCIÓN DE CARGA INTELIGENTE ---
def cargar_archivo(uploaded_file, header_row=0):
    """Detecta si es CSV o Excel y lo carga correctamente."""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                return pd.read_csv(uploaded_file, header=header_row, encoding='utf-8')
            except:
                uploaded_file.seek(0) # Reiniciar puntero si falla el primero
                return pd.read_csv(uploaded_file, header=header_row, encoding='latin1')
        else:
            # Asumimos Excel
            return pd.read_excel(uploaded_file, header=header_row)
    except Exception as e:
        st.error(f"Error cargando {uploaded_file.name}: {e}")
        return None

# --- CLASE PRINCIPAL ---
class CommunityDigitalTwin:
    def __init__(self):
        self.profiles_prob = None
        self.global_prob = None
        self.synthetic_population = None

    def fit(self, df_encuesta):
        # Búsqueda inteligente de columnas
        col_rama = next((c for c in df_encuesta.columns if 'rama' in str(c).lower()), None)
        col_nps = next((c for c in df_encuesta.columns if 'satisfacción' in str(c)), None)
        cols_text = [c for c in df_encuesta.columns if 'Por qué' in str(c) or 'sugerencias' in str(c)]
        
        if not col_rama or not col_nps:
            return False

        # Procesamiento
        df_encuesta['Texto_Full'] = df_encuesta[cols_text].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        def assign_profile(row):
            t = row['Texto_Full']
            n = pd.to_numeric(row[col_nps], errors='coerce')
            if 'cancha' in t or 'luz' in t or 'baño' in t: return 'Crítico Infra'
            if 'ganar' in t or 'compet' in t or 'torneo' in t: return 'Competitivo'
            if 'social' in t or 'amigos' in t: return 'Social'
            if 'profe' in t or 'clase' in t: return 'Formativo'
            if n >= 6: return 'Satisfecho'
            return 'Neutro'

        df_encuesta['Perfil'] = df_encuesta.apply(assign_profile, axis=1)
        self.profiles_prob = pd.crosstab(df_encuesta[col_rama], df_encuesta['Perfil'], normalize='index')
        self.global_prob = df_encuesta['Perfil'].value_counts(normalize=True)
        return True

    def generate(self, df_universe):
        if self.profiles_prob is None: return None
        
        # Mapeo de columnas
        cols_map = {}
        for c in df_universe.columns:
            if 'ramas' in str(c).lower() or 'disciplina' in str(c).lower(): cols_map[c] = 'Group'
            if 'total' in str(c).lower(): cols_map[c] = 'Quantity'
        df_universe.rename(columns=cols_map, inplace=True)
        
        if 'Group' not in df_universe.columns: # Fallback por posición
            try:
                df_universe.rename(columns={df_universe.columns[1]: 'Group', df_universe.columns[4]: 'Quantity'}, inplace=True)
            except: pass

        pop = []
        for _, row in df_universe.iterrows():
            try:
                group = str(row['Group'])
                # Limpieza robusta de números
                qty_raw = str(row['Quantity'])
                if qty_raw.lower() == 'nan' or qty_raw.strip() == '': continue
                
                qty = int(float(qty_raw.replace('.','').replace(',','')))
                
                if qty > 0:
                    match = [i for i in self.profiles_prob.index if str(i).lower() in group.lower()]
                    dist = self.profiles_prob.loc[match[0]] if match else self.global_prob
                    clones = np.random.choice(dist.index, size=qty, p=dist.values)
                    for c in clones: pop.append({'Group': group, 'Profile': c})
            except: continue
        
        self.synthetic_population = pd.DataFrame(pop)
        return self.synthetic_population

# --- INTERFAZ ---
col1, col2 = st.columns(2)
with col1:
    file_encuesta = st.file_uploader("1. Subir Encuesta (CSV/Excel)", type=['csv', 'xlsx'])
with col2:
    file_censo = st.file_uploader("2. Subir Censo (CSV/Excel)", type=['csv', 'xlsx'])

if file_encuesta and file_censo:
    st.info("Procesando archivos...")
    bot = CommunityDigitalTwin()
    
    try:
        # Usamos la función inteligente para cargar
        # Para la encuesta probamos header 0
        df_e = cargar_archivo(file_encuesta, header_row=0)
        
        # Para el censo, a veces el header está en la fila 1 (index 1)
        # Probamos cargar y si falla buscamos la cabecera
        df_c = cargar_archivo(file_censo, header_row=1)
        if df_c is not None and 'Disciplina' not in str(df_c.columns):
             # Si no se ve bien, intentamos con header 0
             file_censo.seek(0)
             df_c = cargar_archivo(file_censo, header_row=0)

        if df_e is not None and df_c is not None:
            # Buscamos la fila correcta en la encuesta si es necesario
            if 'Id respuesta' not in df_e.columns:
                # Buscar dónde está el ID
                for i, row in df_e.head(20).iterrows():
                    if row.astype(str).str.contains('Id respuesta').any():
                        df_e.columns = df_e.iloc[i]
                        df_e = df_e[i+1:].reset_index(drop=True)
                        break

            if bot.fit(df_e):
                df_final = bot.generate(df_c)
                
                if df_final is not None and not df_final.empty:
                    st.divider()
                    st.subheader("🚀 Mapa Estratégico Generado")
                    
                    # Gráfico
                    res = pd.crosstab(df_final['Group'], df_final['Profile'], normalize='index') * 100
                    res['Size'] = df_final['Group'].value_counts()
                    if 'Social' not in res.columns: res['Social'] = 0
                    if 'Competitivo' not in res.columns: res['Competitivo'] = 0
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=res, x='Social', y='Competitivo', size='Size', sizes=(100, 1000), alpha=0.7, ax=ax)
                    for i in range(len(res)):
                        ax.text(res['Social'].iloc[i], res['Competitivo'].iloc[i], res.index[i], size=9)
                    ax.set_title("Mapa de Comunidades")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Descarga
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar CSV Sintético", data=csv, file_name="cduc_sintetico.csv")
                else:
                    st.warning("No se pudo generar la población. Verifica que el archivo de Censo tenga las columnas de 'Disciplina' y 'Total'.")
            else:
                st.error("Error: No se encontraron las columnas clave (Rama, Satisfacción) en la Encuesta.")
    except Exception as e:
        st.error(f"Error técnico detallado: {e}")
