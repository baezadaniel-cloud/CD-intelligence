import streamlit as st
import pandas as pd
import polars as pl  # Motor de Big Data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import io

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(page_title="Plataforma de Análisis con IA", layout="wide", page_icon="🧠")

# --- 2. FUNCIONES DE BACKEND (Email, DB, Carga) ---

def enviar_correo_automatico(destinatario, nombre, metricas):
    """Envía un resumen ejecutivo al correo del Lead."""
    if "EMAIL_USER" not in st.secrets:
        return False # Si no hay secretos configurados, saltamos silenciosamente
        
    sender = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASSWORD"]
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = destinatario
    msg['Subject'] = f"🚀 Reporte de Análisis IA - {nombre}"

    body = f"""
    Hola {nombre},
    
    Tu análisis de comunidad ha finalizado exitosamente en nuestra Plataforma de IA.
    
    RESUMEN EJECUTIVO:
    ------------------
    - Total de Perfiles Analizados: {metricas['total']}
    - Segmento Dominante: {metricas['top_segmento']} ({metricas['pct_dominante']}%)
    - Oportunidad Detectada: {metricas['oportunidad']}
    
    Este es un análisis preliminar basado en Clusterización Psicográfica.
    Para implementar estrategias de fidelización basadas en estos datos, agenda una reunión con nosotros.
    
    Atte,
    Tu Consultora de Inteligencia de Datos.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, destinatario, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error email: {e}")
        return False

def guardar_lead_sheets(datos):
    """Guarda el Lead en Google Sheets (CRM Intermedio)."""
    if "GCP_SERVICE_ACCOUNT" not in st.secrets:
        return False # Modo demo sin DB
        
    try:
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["GCP_SERVICE_ACCOUNT"]), scope)
        client = gspread.authorize(creds)
        # Abre la hoja (Asegúrate de compartirla con el email del bot)
        sheet = client.open("DB_Leads_IA").sheet1
        sheet.append_row([str(datetime.now()), datos['nombre'], datos['apellido'], datos['email'], datos['whatsapp'], "Login Exitoso"])
        return True
    except Exception as e:
        print(f"Error DB: {e}")
        return False

def carga_inteligente(file):
    """
    Usa POLARS para .csv y .parquet (Big Data).
    Usa PANDAS para Excel (Compatibilidad).
    """
    try:
        if file.name.endswith('.parquet'):
            # Polars es nativo y ultra rápido para Parquet
            return pl.read_parquet(file).to_pandas() 
        elif file.name.endswith('.csv'):
            try:
                # Intentamos Polars primero (más rápido)
                return pl.read_csv(file, ignore_errors=True).to_pandas()
            except:
                # Fallback a Pandas si falla
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else:
            # Excel siempre con Pandas
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return None

# --- 3. CLASE MAESTRA (LOGIC CORE) ---
# Aquí vive tu lógica de Segmentación y Nichos
class CommunityDigitalTwin:
    def __init__(self):
        self.profiles_prob = None
        self.global_prob = None
        self.synthetic_population = None

    def fit(self, df, col_rama, col_nps, cols_text):
        # 1. Procesamiento de Texto (NLP Básico)
        df['Texto_Full'] = df[cols_text].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        # 2. Motor de Segmentación (Tus Reglas de Negocio)
        def assign_profile(row):
            t = row['Texto_Full']
            n = pd.to_numeric(row[col_nps], errors='coerce')
            
            # Nichos Específicos
            if 'cancha' in t or 'luz' in t or 'baño' in t: return 'Crítico Infra'
            if 'ganar' in t or 'compet' in t or 'torneo' in t: return 'Competitivo'
            if 'social' in t or 'amigos' in t: return 'Social'
            if 'profe' in t or 'clase' in t: return 'Formativo'
            
            # Segmentación por NPS
            if n >= 6: return 'Satisfecho'
            if n <= 4: return 'Detractor'
            return 'Neutro'

        df['Perfil'] = df.apply(assign_profile, axis=1)
        
        # 3. Modelado Probabilístico (Clusters)
        self.profiles_prob = pd.crosstab(df[col_rama], df['Perfil'], normalize='index')
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
                    # Inferencia Estadística
                    match = [i for i in self.profiles_prob.index if str(i).lower() in group.lower()]
                    dist = self.profiles_prob.loc[match[0]] if match else self.global_prob
                    clones = np.random.choice(dist.index, size=qty, p=dist.values)
                    for c in clones: pop.append({'Group': group, 'Profile': c})
            except: continue
        self.synthetic_population = pd.DataFrame(pop)
        return self.synthetic_population

# --- 4. INTERFAZ DE USUARIO (FRONTEND) ---

# GESTIÓN DE SESIÓN
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    # --- PANTALLA DE LOGIN (LEAD GEN) ---
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔒 Acceso a Plataforma IA")
        st.markdown("Ingresa tus datos para desbloquear el motor de análisis predictivo.")
        
        with st.form("login_form"):
            nombre = st.text_input("Nombre")
            apellido = st.text_input("Apellido")
            email = st.text_input("Correo Corporativo")
            whatsapp = st.text_input("WhatsApp (Opcional - Para alertas)")
            
            if st.form_submit_button("🚀 INGRESAR AL DASHBOARD", type="primary"):
                if nombre and email:
                    # Guardamos Lead
                    lead_data = {'nombre': nombre, 'apellido': apellido, 'email': email, 'whatsapp': whatsapp}
                    guardar_lead_sheets(lead_data)
                    
                    # Iniciamos Sesión
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = lead_data
                    st.rerun()
                else:
                    st.error("Nombre y Correo son obligatorios.")

else:
    # --- PANTALLA PRINCIPAL (APP) ---
    st.sidebar.success(f"Usuario: {st.session_state['user']['nombre']} {st.session_state['user']['apellido']}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.title("🧠 Plataforma de Análisis con IA de Comunidades")
    st.markdown("Genera Gemelos Digitales, detecta nichos ocultos y recibe insights estratégicos.")

    # CARGA DE ARCHIVOS
    c1, c2 = st.columns(2)
    f_encuesta = c1.file_uploader("1. Encuesta (CSV, Excel, Parquet)", type=['csv','xlsx','parquet'])
    f_censo = c2.file_uploader("2. Censo/Universo (CSV, Excel, Parquet)", type=['csv','xlsx','parquet'])

    if f_encuesta and f_censo:
        df_e = carga_inteligente(f_encuesta)
        df_c = carga_inteligente(f_censo)
        
        if df_e is not None and df_c is not None:
            # SELECTORES DE COLUMNAS (Human-in-the-loop)
            st.divider()
            with st.expander("⚙️ Configuración de Variables (Abre si necesitas ajustar)", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Variables Psicométricas (Encuesta)")
                    col_rama = st.selectbox("Columna Segmento/Rama", df_e.columns)
                    col_nps = st.selectbox("Columna Satisfacción (NPS)", df_e.columns)
                    cols_text = st.multiselect("Columnas de Texto (NLP)", df_e.columns)
                with col2:
                    st.caption("Variables Demográficas (Censo)")
                    col_group = st.selectbox("Columna Grupo/Rama", df_c.columns)
                    col_qty = st.selectbox("Columna Cantidad Total", df_c.columns)

            # EJECUCIÓN
            if st.button("✨ Ejecutar Análisis de IA", type="primary"):
                bot = CommunityDigitalTwin()
                
                if cols_text and col_rama:
                    with st.spinner("Procesando Algoritmos de Clusterización..."):
                        bot.fit(df_e, col_rama, col_nps, cols_text)
                        df_final = bot.generate(df_c, col_group, col_qty)
                    
                    if df_final is not None:
                        # --- DASHBOARD DE RESULTADOS ---
                        st.divider()
                        
                        # 1. KPIs
                        top_segment = df_final['Profile'].mode()[0]
                        pct_top = int((df_final['Profile'].value_counts(normalize=True)[0])*100)
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Población Sintética Generada", f"{len(df_final):,} socios")
                        k2.metric("Nicho Dominante", top_segment)
                        k3.metric("Intensidad del Nicho", f"{pct_top}%")

                        # 2. GRÁFICO ESTRATÉGICO
                        st.subheader("📍 Mapa Estratégico de Comunidades")
                        
                        res = pd.crosstab(df_final['Group'], df_final['Profile'], normalize='index') * 100
                        res['Size'] = df_final['Group'].value_counts()
                        if 'Social' not in res.columns: res['Social'] = 0
                        if 'Competitivo' not in res.columns: res['Competitivo'] = 0
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=res, x='Social', y='Competitivo', size='Size', sizes=(100, 1000), hue=res.index, alpha=0.7, ax=ax, legend=False)
                        
                        # Etiquetas inteligentes
                        for i in range(len(res)):
                            ax.text(res['Social'].iloc[i], res['Competitivo'].iloc[i]+2, res.index[i], ha='center', size=8, weight='bold')
                            
                        ax.set_xlabel("% Interés Social (Comunidad)")
                        ax.set_ylabel("% Interés Competitivo (Rendimiento)")
                        ax.axhline(50, ls='--', color='grey', alpha=0.3)
                        ax.axvline(50, ls='--', color='grey', alpha=0.3)
                        st.pyplot(fig)

                        # 3. ENVÍO DE CORREO AUTOMÁTICO
                        metricas = {'total': len(df_final), 'top_segmento': top_segment, 'pct_dominante': pct_top, 'oportunidad': 'Activación Digital'}
                        envio = enviar_correo_automatico(st.session_state['user']['email'], st.session_state['user']['nombre'], metricas)
                        
                        if envio:
                            st.toast("✅ Informe enviado a tu correo exitosamente.")
                        
                        # 4. DESCARGA
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df_final.to_excel(writer, sheet_name='Data_Sintetica', index=False)
                            res.to_excel(writer, sheet_name='Matriz_Estrategica')
                        
                        st.download_button(
                            label="📥 Descargar Reporte Completo (Excel)",
                            data=buffer,
                            file_name="Reporte_IA_Comunidades.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        
                else:
                    st.warning("Selecciona al menos una columna de texto para el análisis.")
