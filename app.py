import streamlit as st
import pandas as pd

# Título simple para confirmar que el código se actualizó
st.title("🚑 MODO DIAGNÓSTICO")

# 1. Prueba de Librerías (Si esto falla, saldrá un error en pantalla)
try:
    import polars as pl
    st.success("✅ Librería Polars instalada correctamente.")
except ImportError:
    st.error("❌ ERROR CRÍTICO: 'polars' no está en requirements.txt")

try:
    import xlsxwriter
    st.success("✅ Librería XlsxWriter instalada correctamente.")
except ImportError:
    st.error("❌ ERROR CRÍTICO: 'xlsxwriter' no está en requirements.txt")

st.write("---")

# 2. Prueba de Carga Básica
uploaded_file = st.file_uploader("Sube tu archivo aquí (Prueba de vida)", type=['xlsx', 'csv', 'parquet'])

if uploaded_file is not None:
    st.info("📡 Archivo recibido por el servidor...")
    st.write(f"📂 Nombre: `{uploaded_file.name}`")
    st.write(f"⚖️ Peso: `{uploaded_file.size} bytes`")

    # Intentamos leer SIN lógica compleja
    try:
        st.write("⏳ Intentando abrir con Pandas básico...")
        
        # Lógica tonta pero segura para probar lectura
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ ¡LECTURA EXITOSA! Filas detectadas: {len(df)}")
        st.dataframe(df.head(5))
        
    except Exception as e:
        st.error(f"❌ El archivo llegó, pero falló al abrirse: {e}")

else:
    st.warning("Esperando archivo...")
