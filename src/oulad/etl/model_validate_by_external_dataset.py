# ==============================================================================
# SCRIPT ADAPTADO: ANÁLISIS USANDO LA ABT SINTÉTICA
# ==============================================================================

# --- LIBRERÍAS REQUERIDAS ---
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from oulad.etl.model import StudentOutcomeModel

print(">>> INICIANDO SCRIPT DE ANÁLISIS CON DATA SINTÉTICA <<<")

# --- ETAPA 1: Carga de la Tabla Analítica Base (ABT) Sintética ---
print("\n--- [ETAPA 1/4] Cargando el archivo 'analytical_base_table_sintetica.csv'...")
try:
    # Ahora cargamos un único archivo que ya contiene todo procesado
    abt_sintetico = pd.read_csv('data/analytical_base_table_sintetica.csv')
    print("ABT Sintética cargada exitosamente.")
    print(f"La tabla tiene {abt_sintetico.shape[0]} filas y {abt_sintetico.shape[1]} columnas.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'analytical_base_table_sintetica.csv'.")
    print("Asegúrate de ejecutar primero el script 'generar_abt_sintetica.py'.")
    exit()

# --- ETAPA 2: Preparación de Datos para el Modelo ---
# OJO: La ingeniería de características ya no es necesaria, pasamos directo a la preparación.
print("\n--- [ETAPA 2/4] Preparando X_sintetico e y_sintetico...")

# Definir columnas de características y objetivo
target_column = 'is_at_risk'
# Excluimos los IDs y la columna 'final_result' original, además del propio target
excluded_cols = ['id_student', 'code_module', 'code_presentation', 'final_result', 'is_at_risk']
# Seleccionamos todas las columnas que no están en la lista de exclusión
# Esta es una forma robusta de asegurarse de que todas las features se incluyan
feature_columns = [col for col in abt_sintetico.columns if col not in excluded_cols]

X_sintetico = abt_sintetico[feature_columns]
y_sintetico = abt_sintetico[target_column]
print("Datos X e y listos para los experimentos.")


# --- ETAPA 3: EXPERIMENTO 1 - Probar la Universalidad del Modelo OULAD ---
print("\n" + "="*80)
print("          EXPERIMENTO 1: ¿Son universales los patrones de OULAD?")
print("="*80 + "\n")
try:
    modelo_oulad = joblib.load('modelo_experto_OULADRegresion_Logistica.pkl')
    print("Modelo 'experto_OULAD' cargado. Realizando predicciones en datos sintéticos...")
    
    # Asegurarnos que las columnas están en el mismo orden que el modelo espera
    # (Si el modelo de OULAD se guardó con nombres de features)
    if hasattr(modelo_oulad, 'feature_name_'):
        X_sintetico_reordenado = X_sintetico[modelo_oulad.feature_name_]
    else:
        X_sintetico_reordenado = X_sintetico

    predicciones_oulad_en_sintetico = modelo_oulad.model.predict(X_sintetico_reordenado)
    
    print("\n--- Resultados del Modelo de OULAD aplicado a los datos sintéticos ---")
    print(classification_report(y_sintetico, predicciones_oulad_en_sintetico, target_names=['No en Riesgo', 'En Riesgo']))
    
except FileNotFoundError:
    print("ADVERTENCIA: No se encontró el archivo 'modelo_experto_OULAD.pkl'. Omitiendo Experimento 1.")
except Exception as e:
    print(f"Ocurrió un error en el Experimento 1: {e}")


# --- ETAPA 4: EXPERIMENTO 2 - Entrenar y Evaluar un Modelo Especialista para los Datos Sintéticos ---
print("\n" + "="*80)
print("          EXPERIMENTO 2: Entrenar el mejor modelo para los datos sintéticos")
print("="*80 + "\n")

print("Dividiendo datos sintéticos para entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X_sintetico, y_sintetico, test_size=0.25, random_state=42, stratify=y_sintetico)

print(f"Distribución de clases ANTES de SMOTE en el set de entrenamiento:\n{y_train.value_counts()}")
print("Aplicando SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Distribución de clases DESPUÉS de SMOTE:\n{y_train_resampled.value_counts()}")

print("\nEntrenando nuevo modelo 'especialista_sintetico'...")
modelo_sintetico = LGBMClassifier(random_state=42)
modelo_sintetico.fit(X_train_resampled, y_train_resampled)

print("Realizando predicciones con el nuevo modelo especialista...")
y_pred_sintetico = modelo_sintetico.predict(X_test)

print("\n--- Resultados del Modelo Especialista entrenado con los datos sintéticos ---")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sintetico).ravel()
confusion_df = pd.DataFrame(
    data=[[tn, fp], [fn, tp]],
    columns=['Predicción: No en Riesgo', 'Predicción: En Riesgo'],
    index=['Real: No en Riesgo', 'Real: En Riesgo']
)
print("Matriz de Confusión:")
print(confusion_df)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_sintetico, target_names=['No en Riesgo', 'En Riesgo']))

print("\n\n>>> SCRIPT FINALIZADO <<<")