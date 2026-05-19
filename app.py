import io                     
import warnings                    
import numpy as np                
import pandas as pd                
import matplotlib.pyplot as plt    
import seaborn as sns              
import plotly.express as px        
import plotly.graph_objects as go  
import streamlit as st             

# Desactivar advertencias para mantener la salida limpia
warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURACIÓN DE LA PÁGINA
# ============================================================

# Configuración general de la app (título, ícono, layout)
st.set_page_config(
    page_title="Telco Churn EDA",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar la apariencia visual
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #0f1b2d, #1a2f4a); }
    [data-testid="stSidebar"] * { color: #e0e7ef !important; }
    div[data-testid="metric-container"] {
        background: #f0f6ff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    .insight-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 0.85rem 1.2rem;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
        color: #1e3a5f;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
#  CLASE PRINCIPAL — DataAnalyzer (POO)
# ============================================================

# Clase que encapsula el análisis, estadísticas y visualizaciones del dataset
class DataAnalyzer:

    # Constructor: recibe el DataFrame y lo almacena como atributo
    def __init__(self, df):
        self.df = df

    # Método para clasificar variables en numéricas y categóricas
    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(include='object').columns.tolist()
        return numericas, categoricas

    # Método para calcular estadísticas descriptivas con mediana, moda y asimetría
    def estadisticas_descriptivas(self):
        numericas, _ = self.clasificar_variables()
        stats = self.df[numericas].describe().T
        stats['median'] = self.df[numericas].median()
        stats['moda'] = self.df[numericas].mode().iloc[0]
        stats['asimetria'] = self.df[numericas].skew()
        return stats.round(2)

    # Método para contar valores nulos por columna con su porcentaje
    def valores_nulos(self):
        total = self.df.isnull().sum()
        porcentaje = (total / len(self.df) * 100).round(2)
        return pd.DataFrame({'Nulos': total, 'Porcentaje (%)': porcentaje})[total > 0]

    # Método para obtener la distribución de la variable objetivo Churn
    def distribucion_churn(self):
        return self.df['Churn'].value_counts()

    # Método para capturar df.info() como texto y mostrarlo en la app
    def info_general(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    # Método para graficar histograma de variable numérica (Plotly)
    def grafico_histograma(self, col):
        fig = px.histogram(
            self.df, x=col, nbins=30,
            color_discrete_sequence=['#2563eb'],
            title=f'Distribución de {col}'
        )
        fig.update_layout(bargap=0.1)
        return fig

    # Método para graficar conteo de variable categórica en barras horizontales (Plotly)
    def grafico_barras(self, col):
        counts = self.df[col].value_counts().reset_index()
        counts.columns = [col, 'Cantidad']
        fig = px.bar(
            counts, x='Cantidad', y=col,
            orientation='h',
            color_discrete_sequence=['#2563eb'],
            title=f'Distribución de {col}'
        )
        return fig

    # Método para graficar boxplot de variable numérica vs Churn (Plotly)
    def grafico_numerico_vs_churn(self, col):
        fig = px.box(
            self.df, x='Churn', y=col,
            color='Churn',
            color_discrete_map={'No': '#2563eb', 'Yes': '#ef4444'},
            title=f'{col} vs Churn'
        )
        return fig

    # Método para graficar tasa de Churn por variable categórica en barras agrupadas (Plotly)
    def grafico_categorico_vs_churn(self, col):
        ct = pd.crosstab(self.df[col], self.df['Churn'], normalize='index') * 100
        ct = ct.reset_index()
        fig = px.bar(
            ct, x=col, y=['No', 'Yes'],
            barmode='group',
            color_discrete_map={'No': '#2563eb', 'Yes': '#ef4444'},
            title=f'Tasa de Churn por {col} (%)',
            labels={'value': 'Porcentaje (%)'}
        )
        return fig

    # Método para graficar heatmap de correlación entre variables numéricas (Seaborn)
    def grafico_correlacion(self):
        numericas, _ = self.clasificar_variables()
        corr = self.df[numericas].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
                    linewidths=0.5, ax=ax, annot_kws={'size': 11})
        ax.set_title('Matriz de Correlación', fontsize=13, fontweight='bold')
        fig.tight_layout()
        return fig

    # Método para graficar pie chart de distribución de Churn (Plotly)
    def grafico_pie_churn(self):
        dist = self.distribucion_churn()
        fig = px.pie(
            values=dist.values,
            names=dist.index,
            color=dist.index,
            color_discrete_map={'No': '#2563eb', 'Yes': '#ef4444'},
            title='Distribución de Churn'
        )
        return fig


# ============================================================
#  FUNCIÓN AUXILIAR — Mostrar caja de insight
# ============================================================

# Función para mostrar un insight destacado con estilo visual
def mostrar_insight(texto):
    st.markdown(f'<div class="insight-box">💡 {texto}</div>', unsafe_allow_html=True)


# ============================================================
#  SIDEBAR — MENÚ DE NAVEGACIÓN
# ============================================================

# Menú lateral con las cuatro secciones principales de la app
with st.sidebar:
    st.markdown("## 📡 Telco Churn EDA")
    st.markdown("---")
    modulo = st.radio(
        "Selecciona un módulo:",
        [
            "🏠 Home",
            "📂 Carga del Dataset",
            "🔍 Análisis Exploratorio (EDA)",
            "📝 Conclusiones"
        ]
    )
    st.markdown("---")
    st.caption("Especialización Python for Analytics · 2026")


# ============================================================
#  ESTADO DE SESIÓN — Guardar datos entre módulos
# ============================================================

# Inicializar variables de sesión si no existen aún
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = None


# ============================================================
#  MÓDULO 1: HOME
# ============================================================

if modulo == "🏠 Home":

    # Título principal de la aplicación
    st.title("📡 Análisis Exploratorio de Datos")
    st.subheader("Telco Customer Churn")
    st.markdown("**Identificando patrones de fuga de clientes mediante análisis visual y estadístico.**")
    st.markdown("---")

    # Dos columnas: objetivo y autor a la izquierda, dataset y tecnologías a la derecha
    col1, col2 = st.columns(2)

    with col1:
        # Objetivo del proyecto
        st.write("### 🎯 Objetivo del Proyecto")
        st.markdown("""
        Este proyecto desarrolla una herramienta interactiva de **Análisis Exploratorio de Datos (EDA)**
        sobre el dataset *Telco Customer Churn*, con el fin de:
        - Comprender la estructura y calidad de los datos
        - Identificar variables asociadas a la fuga de clientes
        - Visualizar patrones que apoyen la toma de decisiones

        > ⚠️ **No se construyen modelos predictivos.** El enfoque es puramente exploratorio.
        """)

        st.write("### 👤 Datos del Autor")
        st.info("""
        **Nombre:** Carla Sarmiento Prada
        **Especialización:** Python for Analytics
        **Año:** 2026
        """)

    with col2:
        # Descripción del dataset utilizado
        st.write("### 📊 Sobre el Dataset")
        st.markdown("""
        El archivo **TelcoCustomerChurn.csv** contiene información de aproximadamente 7,000 clientes
        de una empresa de telecomunicaciones:

        | Categoría | Variables |
        |---|---|
        | Demografía | `gender`, `SeniorCitizen`, `Partner` |
        | Servicios | `PhoneService`, `InternetService`… |
        | Contrato | `Contract`, `PaymentMethod` |
        | Facturación | `MonthlyCharges`, `TotalCharges`, `tenure` |
        | **Objetivo** | **`Churn`** (Yes / No) |
        """)

        # Tecnologías utilizadas en el proyecto
        st.write("### 🛠️ Tecnologías Utilizadas")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("**Python**")
            st.markdown("**Pandas**")
        with col_t2:
            st.markdown("**NumPy**")
            st.markdown("**Plotly**")
            st.markdown("**Matplotlib**")
        with col_t3:
            st.markdown("**Seaborn**")
            st.markdown("**Streamlit**")


# ============================================================
#  MÓDULO 2: CARGA DEL DATASET
# ============================================================

elif modulo == "📂 Carga del Dataset":

    st.title("📂 Carga del Dataset")
    st.markdown("Sube el archivo **TelcoCustomerChurn** en formato `.csv` o `.xlsx` para comenzar el análisis.")
    st.markdown("---")

    # Si ya hay datos cargados, mostrar confirmación
    if st.session_state['df'] is not None:
        df_actual = st.session_state['df']
        st.success("✅ Ya tienes un archivo cargado y listo para analizar.")
        st.info(f"📊 Dataset activo: **{df_actual.shape[0]:,} filas × {df_actual.shape[1]} columnas**")

        # Botón para reemplazar el archivo
        if st.button("🔄 Cargar un archivo diferente"):
            st.session_state['df'] = None
            st.session_state['analyzer'] = None
            st.rerun()

    else:
        # Solo muestra el uploader si no hay archivo cargado aún
        uploaded_file = st.file_uploader("Selecciona el archivo CSV o Excel", type=["csv", "xlsx"])

        if uploaded_file is not None:
            nombre = uploaded_file.name

            if nombre.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Error al leer el archivo CSV: {e}")
                    st.stop()

            elif nombre.endswith(".xlsx"):
                try:
                    df = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Error al leer el archivo Excel: {e}")
                    st.stop()

            else:
                st.warning(
                    f"⚠️ El archivo **'{nombre}'** no es compatible. "
                    "Por favor sube un archivo en formato **.csv** o **.xlsx**."
                )
                st.stop()

            try:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

                analyzer = DataAnalyzer(df)
                st.session_state['df'] = df
                st.session_state['analyzer'] = analyzer

                st.success(f"✅ Archivo cargado correctamente: **{nombre}**")

                st.write("### Vista Previa del Dataset")
                n_filas = st.slider("Número de filas a mostrar", 5, 20, 5)
                st.write(df.head(n_filas))

                st.write("### Dimensiones del Dataset")
                col1, col2, col3 = st.columns(3)
                col1.metric("🧾 Filas", f"{df.shape[0]:,}")
                col2.metric("📋 Columnas", df.shape[1])
                col3.metric("🔴 Clientes con Churn", f"{(df['Churn'] == 'Yes').sum():,}")

                st.write("### Tipos de Datos")
                tipo_df = pd.DataFrame({
                    'Columna': df.dtypes.index,
                    'Tipo': df.dtypes.values.astype(str),
                    'Nulos': df.isnull().sum().values,
                    'Únicos': df.nunique().values
                })
                st.write(tipo_df)

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

        else:
            st.warning("⚠️ Por favor carga el archivo CSV o Excel para continuar.")


# ============================================================
#  MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================

elif modulo == "🔍 Análisis Exploratorio (EDA)":

    # Título del módulo
    st.title("🔍 Análisis Exploratorio de Datos")

    # Verificar que el dataset fue cargado antes de ejecutar el análisis
    if st.session_state['df'] is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **📂 Carga del Dataset**.")
        st.stop()

    # Recuperar el DataFrame y el analizador desde la sesión
    df = st.session_state['df']
    az = st.session_state['analyzer']

    # Organizar los 10 ítems del EDA en tabs separados
    tabs = st.tabs([
        "1 · Info General",
        "2 · Variables",
        "3 · Estadísticas",
        "4 · Nulos",
        "5 · Numéricas",
        "6 · Categóricas",
        "7 · Biv. Numérico",
        "8 · Biv. Categórico",
        "9 · Análisis Dinámico",
        "10 · Hallazgos"
    ])

    # ----------------------------------------------------------
    # TAB 1: Información general del dataset
    # ----------------------------------------------------------
    with tabs[0]:
        st.write("### Ítem 1 — Información General del Dataset")

        # Dos columnas: info() a la izquierda, tabla de tipos a la derecha
        col1, col2 = st.columns(2)

        with col1:
            st.write("**📋 Salida de df.info()**")
            # Mostramos df.info() capturado como texto con io.StringIO
            st.code(az.info_general(), language='text')

        with col2:
            st.write("**🔢 Tipos de datos y valores nulos**")
            tipo_df = pd.DataFrame({
                'Columna': df.dtypes.index,
                'Tipo': df.dtypes.values.astype(str),
                'Nulos': df.isnull().sum().values,
                'Únicos': df.nunique().values
            })
            st.write(tipo_df)

        # Insight principal del ítem
        mostrar_insight("El dataset tiene columnas numéricas y categóricas. TotalCharges puede tener nulos tras la conversión.")

    # ----------------------------------------------------------
    # TAB 2: Clasificación de variables
    # ----------------------------------------------------------
    with tabs[1]:
        st.write("### Ítem 2 — Clasificación de Variables")

        # Llamar al método de la clase para clasificar variables
        numericas, categoricas = az.clasificar_variables()

        # Mostrar listas en dos columnas
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**🔢 Variables Numéricas ({len(numericas)})**")
            for v in numericas:
                st.markdown(f"- `{v}`")

        with col2:
            st.write(f"**🔤 Variables Categóricas ({len(categoricas)})**")
            for v in categoricas:
                st.markdown(f"- `{v}`")

        # Insight principal del ítem
        mostrar_insight(
            f"Se identificaron {len(numericas)} variables numéricas y "
            f"{len(categoricas)} categóricas. La variable objetivo `Churn` es categórica (Yes/No)."
        )

    # ----------------------------------------------------------
    # TAB 3: Estadísticas descriptivas
    # ----------------------------------------------------------
    with tabs[2]:
        st.write("### Ítem 3 — Estadísticas Descriptivas")

        # Mostrar tabla con media, mediana, moda, desviación y asimetría
        st.write(az.estadisticas_descriptivas())

        # Métricas destacadas para las variables más relevantes
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Media MonthlyCharges", f"S/{df['MonthlyCharges'].mean():.2f}")
        col2.metric("📅 Mediana Tenure", f"{df['tenure'].median():.0f} meses")
        col3.metric("💳 Media TotalCharges", f"S/{df['TotalCharges'].mean():,.2f}")

        # Insight principal del ítem
        mostrar_insight(
            "MonthlyCharges presenta distribución asimétrica. "
            "La mediana de tenure (~29 meses) indica que la mitad de los clientes llevan menos de 2.5 años."
        )

    # ----------------------------------------------------------
    # TAB 4: Análisis de valores faltantes
    # ----------------------------------------------------------
    with tabs[3]:
        st.write("### Ítem 4 — Análisis de Valores Faltantes")

        # Obtener tabla de nulos usando el método de la clase
        missing = az.valores_nulos()

        if missing.empty:
            # Mensaje positivo si no hay nulos
            st.success("✅ No se encontraron valores faltantes en el dataset.")
        else:
            # Mostrar tabla y gráfico de barras de nulos
            col1, col2 = st.columns(2)
            with col1:
                st.write(missing)
            with col2:
                fig = px.bar(
                    missing.reset_index(),
                    x='Porcentaje (%)', y='index',
                    orientation='h',
                    color_discrete_sequence=['#ef4444'],
                    title='Valores Faltantes por Columna (%)'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Insight principal del ítem
            mostrar_insight(
                "Los nulos en TotalCharges corresponden a clientes nuevos (tenure=0) "
                "que aún no han generado cargos totales."
            )

    # ----------------------------------------------------------
    # TAB 5: Distribución de variables numéricas
    # ----------------------------------------------------------
    with tabs[4]:
        st.write("### Ítem 5 — Distribución de Variables Numéricas")

        # Obtener lista de variables numéricas
        numericas, _ = az.clasificar_variables()

        # Widget para seleccionar qué variables graficar
        cols_sel = st.multiselect(
            "Selecciona variables a visualizar:",
            numericas,
            default=numericas
        )

        if cols_sel:
            # Mostrar histogramas en pares usando columnas de Streamlit
            for i in range(0, len(cols_sel), 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(az.grafico_histograma(cols_sel[i]), use_container_width=True)
                if i + 1 < len(cols_sel):
                    with col2:
                        st.plotly_chart(az.grafico_histograma(cols_sel[i + 1]), use_container_width=True)

            # Insight principal del ítem
            mostrar_insight(
                "MonthlyCharges muestra distribución bimodal. "
                "Tenure tiene alta concentración en valores bajos, indicando muchos clientes recientes."
            )
        else:
            st.info("Selecciona al menos una variable.")

    # ----------------------------------------------------------
    # TAB 6: Análisis de variables categóricas
    # ----------------------------------------------------------
    with tabs[5]:
        st.write("### Ítem 6 — Análisis de Variables Categóricas")

        # Obtener variables categóricas y excluir el ID
        _, categoricas = az.clasificar_variables()
        categoricas = [c for c in categoricas if c != 'customerID']

        # Selectbox para elegir la variable y tabla + gráfico en columnas
        col1, col2 = st.columns([1, 2])
        with col1:
            cat_sel = st.selectbox("Selecciona variable categórica:", categoricas)

            # Tabla con conteos y proporciones de cada categoría
            counts = df[cat_sel].value_counts()
            tabla = pd.DataFrame({
                'Valor': counts.index,
                'Conteo': counts.values,
                'Proporción (%)': (counts.values / len(df) * 100).round(2)
            })
            st.write(tabla)

        with col2:
            # Gráfico de barras horizontal con Plotly
            st.plotly_chart(az.grafico_barras(cat_sel), use_container_width=True)

        # Insight principal del ítem
        mostrar_insight(
            f"La distribución de '{cat_sel}' muestra la proporción de cada categoría sobre el total de clientes."
        )

    # ----------------------------------------------------------
    # TAB 7: Análisis bivariado numérico vs Churn
    # ----------------------------------------------------------
    with tabs[6]:
        st.write("### Ítem 7 — Análisis Bivariado: Numérico vs Churn")

        # Obtener variables numéricas disponibles
        numericas, _ = az.clasificar_variables()

        # Selectbox para elegir la variable numérica a comparar
        num_sel = st.selectbox("Variable numérica:", numericas, key='biv_num')

        # Boxplot interactivo separado por grupo Churn
        st.plotly_chart(az.grafico_numerico_vs_churn(num_sel), use_container_width=True)

        # Métricas comparativas entre clientes con y sin Churn
        col1, col2 = st.columns(2)
        media_no = df.loc[df['Churn'] == 'No', num_sel].mean()
        media_yes = df.loc[df['Churn'] == 'Yes', num_sel].mean()
        col1.metric(f"Media {num_sel} — No Churn", f"{media_no:.2f}")
        col2.metric(f"Media {num_sel} — Churn", f"{media_yes:.2f}")

        # Calcular y mostrar la diferencia porcentual entre grupos
        diff_pct = abs(media_yes - media_no) / media_no * 100
        mostrar_insight(
            f"Los clientes con Churn presentan una diferencia del {diff_pct:.1f}% "
            f"en `{num_sel}` respecto a los que permanecen."
        )

    # ----------------------------------------------------------
    # TAB 8: Análisis bivariado categórico vs Churn
    # ----------------------------------------------------------
    with tabs[7]:
        st.write("### Ítem 8 — Análisis Bivariado: Categórico vs Churn")

        # Obtener variables categóricas, excluyendo ID y la variable objetivo
        _, categoricas = az.clasificar_variables()
        cat_cols = [c for c in categoricas if c not in ['customerID', 'Churn']]

        # Selectbox para elegir la variable categórica a analizar
        cat_biv_sel = st.selectbox("Variable categórica:", cat_cols, key='biv_cat')

        # Gráfico de barras agrupadas y tabla de contingencia en columnas
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(az.grafico_categorico_vs_churn(cat_biv_sel), use_container_width=True)
        with col2:
            # Tabla de contingencia con porcentajes por categoría
            ct = pd.crosstab(df[cat_biv_sel], df['Churn'])
            ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
            st.write("**Tabla de contingencia (%)**")
            st.write(ct_pct)

        # Insight principal del ítem
        mostrar_insight(
            f"El análisis de '{cat_biv_sel}' vs Churn revela diferencias entre categorías. "
            "Contratos mes a mes y fibra óptica tienen las tasas más altas de abandono."
        )

    # ----------------------------------------------------------
    # TAB 9: Análisis dinámico con filtros interactivos
    # ----------------------------------------------------------
    with tabs[8]:
        st.write("### Ítem 9 — Análisis Dinámico Personalizado")
        st.markdown("Configura los filtros y el tipo de análisis que deseas explorar.")

        # Fila de configuración principal con tres widgets
        col1, col2, col3 = st.columns(3)
        with col1:
            # Tipo de análisis a realizar
            tipo_analisis = st.selectbox(
                "Tipo de análisis:",
                ["Distribución univariada", "Correlación numérica",
                 "Numérico vs Churn", "Categórico vs Churn"]
            )
        with col2:
            # Filtro por tipo de contrato (multiselect)
            contratos = df['Contract'].unique().tolist()
            filtro_contrato = st.multiselect(
                "Filtrar por contrato:",
                contratos,
                default=contratos
            )
        with col3:
            # Filtro por rango de meses de permanencia (slider)
            rango_tenure = st.slider(
                "Rango de Tenure (meses):",
                int(df['tenure'].min()),
                int(df['tenure'].max()),
                (0, int(df['tenure'].max()))
            )

        # Checkbox para filtrar solo adultos mayores
        solo_senior = st.checkbox("Incluir solo adultos mayores (SeniorCitizen = Yes)", value=False)

        # Aplicar todos los filtros seleccionados al DataFrame
        df_filtrado = df[
            (df['Contract'].isin(filtro_contrato)) &
            (df['tenure'] >= rango_tenure[0]) &
            (df['tenure'] <= rango_tenure[1])
        ]
        if solo_senior:
            df_filtrado = df_filtrado[df_filtrado['SeniorCitizen'] == 'Yes']

        # Crear un nuevo analizador con los datos filtrados
        az_filtrado = DataAnalyzer(df_filtrado)

        # Mostrar el total de clientes en la selección actual
        st.markdown(f"**Clientes en la selección:** {len(df_filtrado):,}")

        # Obtener variables del DataFrame filtrado
        numericas_f, categoricas_f = az_filtrado.clasificar_variables()

        # Mostrar el gráfico según el tipo de análisis elegido
        if tipo_analisis == "Distribución univariada":
            var = st.selectbox("Variable:", numericas_f, key='dyn_uni')
            st.plotly_chart(az_filtrado.grafico_histograma(var), use_container_width=True, key='chart_dyn_uni')

        elif tipo_analisis == "Correlación numérica":
            # El heatmap usa Seaborn + Matplotlib
            fig = az_filtrado.grafico_correlacion()
            st.pyplot(fig)

        elif tipo_analisis == "Numérico vs Churn":
            var = st.selectbox("Variable:", numericas_f, key='dyn_num')
            st.plotly_chart(az_filtrado.grafico_numerico_vs_churn(var), use_container_width=True, key='chart_dyn_num')

        else:
            cat_f = [c for c in categoricas_f if c not in ['customerID', 'Churn']]
            var = st.selectbox("Variable:", cat_f, key='dyn_cat')
            st.plotly_chart(az_filtrado.grafico_categorico_vs_churn(var), use_container_width=True, key='chart_dyn_cat')

    # ----------------------------------------------------------
    # TAB 10: Hallazgos clave del EDA
    # ----------------------------------------------------------
    with tabs[9]:
        st.write("### Ítem 10 — Hallazgos Clave del EDA")

        # Calcular las métricas principales para el panel resumen
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        media_mc_churn = df.loc[df['Churn'] == 'Yes', 'MonthlyCharges'].mean()
        media_tenure_churn = df.loc[df['Churn'] == 'Yes', 'tenure'].mean()
        pct_month_churn = (
            df.loc[df['Churn'] == 'Yes', 'Contract']
            .value_counts(normalize=True)
            .get('Month-to-month', 0) * 100
        )

        # Panel de métricas en cuatro columnas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉 Tasa de Churn global", f"{churn_rate:.1f}%")
        col2.metric("💰 Cargo mensual (Churn)", f"S/{media_mc_churn:.0f}")
        col3.metric("📅 Tenure promedio (Churn)", f"{media_tenure_churn:.0f} meses")
        col4.metric("📄 Contratos mes a mes (Churn)", f"{pct_month_churn:.0f}%")

        st.markdown("---")

        # Cuatro visualizaciones resumen organizadas en dos columnas
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tasa de Churn por tipo de contrato**")
            st.plotly_chart(az.grafico_categorico_vs_churn('Contract'), use_container_width=True)

            st.write("**Tasa de Churn por servicio de internet**")
            st.plotly_chart(az.grafico_categorico_vs_churn('InternetService'), use_container_width=True)

        with col2:
            st.write("**Proporción global de Churn**")
            st.plotly_chart(az.grafico_pie_churn(), use_container_width=True)

            st.write("**MonthlyCharges vs Churn**")
            st.plotly_chart(az.grafico_numerico_vs_churn('MonthlyCharges'), use_container_width=True)

        st.markdown("---")

        # Insights principales derivados del análisis
        st.write("#### 💡 Principales Insights")
        insights = [
            f"La tasa global de Churn es {churn_rate:.1f}%, concentrada en clientes con contratos mes a mes.",
            "Los clientes con fibra óptica presentan la mayor tasa de abandono entre los tipos de internet.",
            "El cargo mensual de los clientes que abandonan es ~S/13 mayor al de los que permanecen.",
            "Clientes con bajo tenure (<12 meses) son los más propensos al Churn.",
            "La ausencia de TechSupport y OnlineSecurity se correlaciona con mayor fuga."
        ]
        for ins in insights:
            mostrar_insight(ins)


# ============================================================
#  MÓDULO 4: CONCLUSIONES
# ============================================================

elif modulo == "📝 Conclusiones":

    # Título del módulo de conclusiones
    st.title("📝 Conclusiones Finales del Análisis")
    st.markdown("---")

    if st.session_state['df'] is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **📂 Carga del Dataset**.")
        st.stop()

    # Las 5 conclusiones 
    conclusiones = [
        {
            "num": "01",
            "titulo": "El tipo de contrato es el factor más determinante del Churn",
            "cuerpo": (
                "Los clientes con contratos mes a mes presentan una tasa de abandono significativamente "
                "superior a los contratos anuales o bianuales. Incentivar la migración a contratos de "
                "largo plazo es una palanca estratégica de retención de alto impacto."
            )
        },
        {
            "num": "02",
            "titulo": "Los clientes nuevos son el segmento más vulnerable",
            "cuerpo": (
                "La concentración de Churn en los primeros 12 meses de tenure indica una experiencia "
                "inicial deficiente o expectativas no cumplidas. Implementar programas de onboarding "
                "en este período podría reducir significativamente la fuga temprana."
            )
        },
        {
            "num": "03",
            "titulo": "La facturación mensual elevada aumenta el riesgo de abandono",
            "cuerpo": (
                "Los clientes que abandonan pagan en promedio S/13 más por mes que los que permanecen. "
                "Esto puede indicar percepción de bajo valor por el precio pagado, especialmente "
                "si no cuentan con servicios adicionales como soporte técnico o seguridad en línea."
            )
        },
        {
            "num": "04",
            "titulo": "El servicio de fibra óptica requiere atención especial",
            "cuerpo": (
                "El servicio de mayor velocidad concentra la mayor tasa de abandono. Esto puede deberse "
                "a problemas de calidad, precio elevado o mayor competencia en ese segmento. "
                "Se recomienda revisar la propuesta de valor para estos clientes."
            )
        },
        {
            "num": "05",
            "titulo": "Los servicios de valor agregado actúan como ancla de retención",
            "cuerpo": (
                "Clientes con TechSupport, OnlineSecurity y OnlineBackup activos presentan menor "
                "propensión al Churn. La oferta de servicios complementarios no solo genera ingresos "
                "adicionales, sino que incrementa la fidelización del cliente."
            )
        }
    ]

    # Mostrar cada conclusión dentro de un expander expandido
    for c in conclusiones:
        with st.expander(f"**Conclusión {c['num']}** — {c['titulo']}", expanded=True):
            st.markdown(c['cuerpo'])

    st.markdown("---")

    # Recomendaciones de negocio derivadas del análisis
    st.write("### 🚀 Recomendaciones de Negocio")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🎁 **Incentivos para contratos anuales**\n\nDescuentos o beneficios por migrar de contratos mes a mes.")
    with col2:
        st.info("🤝 **Programa de onboarding**\n\nAcompañamiento activo durante los primeros 3 meses del cliente.")
    with col3:
        st.info("🛡️ **Bundling de servicios**\n\nPaquetes con TechSupport + Security a precio especial para aumentar retención.")

    # Pie de página 
    st.markdown("---")
    st.caption("Proyecto desarrollado como parte de la Especialización Python for Analytics · 2026")
