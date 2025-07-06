import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

print("Iniciando la generación de la Tabla Analítica Base (ABT) Sintética...")

# --- 1. Parámetros de la Simulación ---

# La lista definitiva de 35 características que tu modelo de OULAD requiere
feature_columns_definitivas = [
    'gender', 'region', 'education_level', 'poverty_index_band', 'age_band', 
    'has_disability', 'previous_attempts_count', 'studied_credits_total', 
    'registration_day', 'unregistration_day', 'is_unregistered', 
    'days_active_before_unreg', 'assignments_submitted_count', 
    'assignments_avg_score', 'assignments_score_std_dev', 'late_submissions_count', 
    'total_clicks', 'days_engaged_count', 'clicks_content', 'clicks_forum', 
    'clicks_quiz', 'clicks_resource', 'clicks_subpage', 'clicks_homepage', 
    'clicks_questionnaire', 'clicks_ouwiki', 'clicks_htmlactivity', 
    'clicks_ouelluminate', 'clicks_dataplus', 'clicks_externalquiz', 
    'clicks_repeatactivity', 'clicks_dualpane', 'clicks_glossary', 
    'clicks_oucollaborate', 'clicks_folder'
]

n_muestras = 10000 
n_total_features = len(feature_columns_definitivas) # Esto se establece en 35

# Pesos de las clases para simular el desbalance.
# Orden ajustado: [Withdrawn, Fail, Pass, Distinction]
pesos_clases = [0.35, 0.20, 0.35, 0.10] 

# --- 2. Generación de Datos Base con scikit-learn ---
X, y = make_classification(
    n_samples=n_muestras,
    n_features=n_total_features,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    n_clusters_per_class=2,
    weights=pesos_clases,
    flip_y=0.05,
    random_state=42
)

print(f"Datos base generados con {n_total_features} características.")

# --- 3. Construcción del DataFrame (ABT) ---
# Se usan los 35 nombres de la lista para las columnas
abt_sintetico = pd.DataFrame(X, columns=feature_columns_definitivas)

# Añadir columnas de ID y contexto para simular una ABT real
abt_sintetico['id_student'] = [100000 + i for i in range(n_muestras)]
abt_sintetico['code_module'] = 'SYNTH_MOD'
abt_sintetico['code_presentation'] = '2024S'

# --- 4. Crear las Columnas Objetivo ---
# Mapear la variable objetivo numérica 'y' a las etiquetas de texto con la asignación correcta
mapa_resultado = {
    0: 'Withdrawn', # Clase 0
    1: 'Fail',      # Clase 1
    2: 'Pass',
    3: 'Distinction'
}
abt_sintetico['final_result'] = pd.Series(y).map(mapa_resultado)

# Crear la variable objetivo binaria que usará el modelo
abt_sintetico['is_at_risk'] = np.where(abt_sintetico['final_result'] == 'Fail', 1, 0)

print("Columnas de identificación y objetivo añadidas.")

# --- 5. Ajustar los Datos para que Parezcan Reales (Maquillaje Estadístico) ---
# Esto no cambia los patrones subyacentes, solo hace que los rangos se vean más lógicos
abt_sintetico['assignments_avg_score'] = np.abs(abt_sintetico['assignments_avg_score'] * 30 + 65).clip(0, 100)
abt_sintetico['assignments_submitted_count'] = np.abs(abt_sintetico['assignments_submitted_count'] * 4 + 7).astype(int)
abt_sintetico['total_clicks'] = np.abs(abt_sintetico['total_clicks'] * 800 + 1500).astype(int)
abt_sintetico['days_engaged_count'] = np.abs(abt_sintetico['days_engaged_count'] * 50 + 80).astype(int).clip(1, 260)
abt_sintetico['late_submissions_count'] = np.abs(abt_sintetico['late_submissions_count'] * 3).astype(int)
abt_sintetico['registration_day'] = (abt_sintetico['registration_day'] * 25 - 15).astype(int)
abt_sintetico['assignments_score_std_dev'] = np.abs(abt_sintetico['assignments_score_std_dev'] * 10 + 12)
abt_sintetico['studied_credits_total'] = np.random.choice([30, 60, 90, 120], size=n_muestras, p=[0.6, 0.2, 0.1, 0.1])

print("Valores de las características ajustados para mayor realismo.")

# --- 6. Guardar la ABT Sintética en un Archivo CSV ---
nombre_archivo_salida = 'analytical_base_table_sintetica.csv'
print(f"\nGuardando la ABT sintética en '{nombre_archivo_salida}'...")

abt_sintetico.to_csv("data/"+nombre_archivo_salida, index=False)

print("\n¡Proceso finalizado con éxito!")
print(f"Se ha creado el archivo '{nombre_archivo_salida}' con {abt_sintetico.shape[0]} filas y {abt_sintetico.shape[1]} columnas.")
print("\nDistribución de la variable 'is_at_risk' que se generó:")
print(abt_sintetico['is_at_risk'].value_counts(normalize=True))