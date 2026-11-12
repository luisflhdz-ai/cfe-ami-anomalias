import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# Configuración de la página
st.set_page_config(page_title="CFE - Detección de Anomalías AMI", layout="wide")

# 1. Generación y Preprocesamiento de Datos (En caché para optimizar rendimiento)
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000
    horas = np.random.randint(0, 24, n_samples)
    tipo_tarifa = np.random.choice([1, 2], n_samples)
    consumo_kwh = np.random.normal(loc=15, scale=5, size=n_samples)
    consumo_kwh += (horas >= 18) * (horas <= 22) * 10 
    
    n_anomalias = int(n_samples * 0.05)
    idx_anomalias = np.random.choice(n_samples, n_anomalias, replace=False)
    es_anomalia = np.zeros(n_samples)
    es_anomalia[idx_anomalias] = 1
    consumo_kwh[idx_anomalias] = np.random.uniform(0, 2, n_anomalias)
    
    df = pd.DataFrame({
        'hora_del_dia': horas,
        'tipo_tarifa': tipo_tarifa,
        'consumo_kwh': consumo_kwh,
        'es_anomalia': es_anomalia
    })
    return df

@st.cache_resource
def train_model(df):
    X = df[['hora_del_dia', 'tipo_tarifa', 'consumo_kwh']]
    y = df['es_anomalia']
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    smote = SMOTE(random_state=42)
    X_res, _ = smote.fit_resample(X_scaled, y)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_res)
    
    return model, scaler

df = load_data()
model, scaler = train_model(df)

# Título Principal
st.title("⚡ Panel de Control CFE: Detección de Pérdidas No Técnicas")
st.markdown("Sistema de analítica para medidores AMI mediante Isolation Forest.")

# Creación de las 3 secciones requeridas mediante pestañas
tab1, tab2, tab3 = st.tabs(["📊 Visión General", "📈 Exploración Interactiva", "🤖 Simulador / Inferencia"])

# --- SECCIÓN 1: VISIÓN GENERAL ---
with tab1:
    st.header("Visión General de la Red")
    col1, col2, col3 = st.columns(3)
    
    total_medidores = len(df)
    anomalias_detectadas = df['es_anomalia'].sum()
    consumo_promedio = df['consumo_kwh'].mean()
    
    col1.metric("Lecturas Totales Procesadas", f"{total_medidores:,}")
    col2.metric("Alertas Históricas (Fraude/Falla)", f"{int(anomalias_detectadas)}", delta="-15% vs mes anterior", delta_color="inverse")
    col3.metric("Consumo Promedio Global", f"{consumo_promedio:.2f} kWh")
    
    st.dataframe(df.head(10), use_container_width=True)

# --- SECCIÓN 2: EXPLORACIÓN INTERACTIVA ---
with tab2:
    st.header("Análisis de Patrones de Consumo")
    
    # Filtros en la interfaz
    tarifa_filter = st.selectbox("Filtrar por Tipo de Tarifa:", options=["Ambas", "1 (Residencial)", "2 (Comercial)"])
    
    df_filtered = df.copy()
    if tarifa_filter == "1 (Residencial)":
        df_filtered = df_filtered[df_filtered['tipo_tarifa'] == 1]
    elif tarifa_filter == "2 (Comercial)":
        df_filtered = df_filtered[df_filtered['tipo_tarifa'] == 2]

    # Gráfica interactiva
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df_filtered, x='hora_del_dia', y='consumo_kwh', hue='es_anomalia', 
                    palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=ax)
    ax.set_title("Distribución del Consumo por Hora del Día")
    ax.set_xlabel("Hora del Día (0-23)")
    ax.set_ylabel("Consumo (kWh)")
    st.pyplot(fig)

# --- SECCIÓN 3: SIMULADOR / INFERENCIA ---
with tab3:
    st.header("Simulador de Riesgo en Tiempo Real")
    st.markdown("Ingresa los datos de una lectura específica para evaluar si representa una anomalía (posible pérdida no técnica).")
    
    with st.form("simulador_form"):
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            input_hora = st.slider("Hora del Día:", 0, 23, 19)
            input_tarifa = st.selectbox("Tipo de Tarifa:", [1, 2])
        with col_form2:
            input_consumo = st.number_input("Consumo Registrado (kWh):", min_value=0.0, max_value=100.0, value=1.5)
        
        submitted = st.form_submit_button("Evaluar Riesgo")
        
        if submitted:
            # Preparar datos de entrada
            input_data = pd.DataFrame({
                'hora_del_dia': [input_hora],
                'tipo_tarifa': [input_tarifa],
                'consumo_kwh': [input_consumo]
            })
            
            # Escalar y predecir
            input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            pred_raw = model.predict(input_scaled)
            resultado = 1 if pred_raw[0] == -1 else 0
            
            if resultado == 1:
                st.error("🚨 **ALERTA DETECTADA:** El patrón ingresado coincide con una anomalía. Se recomienda inspección en campo por posible fraude o falla del medidor.")
            else:
                st.success("✅ **LECTURA NORMAL:** El consumo se encuentra dentro de los parámetros esperados para esa tarifa y horario.")
