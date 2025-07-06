# =============================================================================
# LIBRERÍAS NECESARIAS
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import lightgbm as lgb # Importamos LightGBM

# Módulos de Scikit-learn para el modelado y la evaluación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    mean_squared_error
)


# =============================================================================
# CLASE PARA EL MODELADO
# =============================================================================
class StudentOutcomeModel:
    """
    Clase flexible para encapsular el entrenamiento, evaluación y exportación 
    de resultados de modelos de clasificación para predecir resultados estudiantiles.
    """
    def __init__(self, model):
        """
        Inicializa la clase con un modelo de scikit-learn.
        Ej: LogisticRegression(), RandomForestClassifier(), lgb.LGBMClassifier()
        """
        if model is None:
            raise ValueError("Se debe proporcionar un modelo de scikit-learn.")
        self.model = model
        self.scaler = StandardScaler()
        self.trained_model = None
        self.trained_scaler = None

    def train(self, df: pd.DataFrame, feature_cols: list, target_col: str, test_size=0.25, random_state=42):
        """
        Prepara los datos, los divide en entrenamiento/prueba y entrena el modelo.
        """
        print(f"--- Iniciando entrenamiento para el modelo: {type(self.model).__name__} ---")
        X = df[feature_cols]
        y = df[target_col]

        print("\n--- DIAGNÓSTICO DE DATOS (ANTES DE LA DIVISIÓN) ---")
        print(f"Analizando la columna objetivo '{target_col}' en el DataFrame completo...")
        print("Distribución de clases en TODO el conjunto de datos:")
        print(y.value_counts())
        print("--------------------------------------------------\n")

        

        

        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print("Escalando características numéricas...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.trained_scaler = self.scaler
        
        print("Entrenando el modelo...")
        self.model.fit(X_train_scaled, y_train)
        self.trained_model = self.model
        print("Modelo entrenado exitosamente.")
        
        # Devuelve todos los conjuntos de datos para su uso posterior
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def evaluate_and_plot(self, X_test_scaled: np.ndarray, y_test: pd.Series):
        """
        Imprime en consola y grafica la evaluación visual del modelo.
        """
        if self.trained_model is None:
            raise RuntimeError("El modelo debe ser entrenado antes de poder ser evaluado.")
            
        print("\n--- Evaluación Visual del Modelo ---")
        y_pred = self.trained_model.predict(X_test_scaled)
        
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=['No en Riesgo (0)', 'En Riesgo (1)']))
        
        print("\nMatriz de Confusión:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No en Riesgo (0)', 'En Riesgo (1)'],
                    yticklabels=['No en Riesgo (0)', 'En Riesgo (1)'])
        plt.title(f'Matriz de Confusión - {type(self.model).__name__}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.show()

    def get_summary_metrics(self, y_test: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Calcula y devuelve un diccionario con las métricas de rendimiento generales.
        Incluye el cálculo manual de F1-Score a partir de TP, FP, TN, FN.
        """
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0

        metrics = {
            'model_name': type(self.model).__name__,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred),
            'mse_on_classification': mean_squared_error(y_test, y_pred), # Nota: MSE no es una métrica estándar para clasificación.
            'true_positives (TP)': tp,
            'false_positives (FP)': fp,
            'true_negatives (TN)': tn,
            'false_negatives (FN)': fn,
            'f1_score_manual': f1_manual
        }
        return metrics

    def get_predictions_df(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Crea y devuelve un DataFrame con los resultados caso a caso.
        """
        predictions_df = X_test.copy()
        predictions_df['true_label'] = y_test
        predictions_df['predicted_label'] = y_pred
        return predictions_df
        
    def plot_feature_importance(self, feature_names: list, top_n=15):
        """
        Visualiza la importancia de las características del modelo.
        """
        if not (hasattr(self.trained_model, 'feature_importances_') or hasattr(self.trained_model, 'coef_')):
            print(f"El modelo {type(self.model).__name__} no soporta la extracción de importancia de características.")
            return

        if hasattr(self.trained_model, 'feature_importances_'):
            importances = self.trained_model.feature_importances_
            title = f'Importancia de Características ({type(self.model).__name__})'
        else:
            importances = self.trained_model.coef_[0]
            title = f'Coeficientes de Características ({type(self.model).__name__})'

        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df['abs_importance'] = feature_importance_df['importance'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='abs_importance', ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title(title, fontsize=16)
        plt.show()

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # 1. CONEXIÓN A LA BASE DE DATOS Y CARGA DE DATOS
    DB_USER = 'oulad_user'
    DB_PASSWORD = 'oulad_pass'
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_NAME = 'oulad_db'
    
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        print("Cargando ABT desde la base de datos...")
        # Asegúrate de que el nombre de la tabla es el final que creaste (ej. 'analytical_base_table_exhaustiva')
        query = "SELECT * FROM analytical_base_table;" 
        df_abt = pd.read_sql(query, con=engine)
        print(f"Carga completa: {len(df_abt)} filas.")
    except Exception as e:
        print(f"ERROR AL CARGAR DATOS: {e}")
        df_abt = pd.DataFrame() # Crear un df vacío para que el script no se caiga

    if not df_abt.empty:
        # 2. PREPARACIÓN FINAL DE LOS DATOS PARA EL MODELO
        print("\n--- Preparando datos para el modelado ---")
        
        # Crear la variable objetivo binaria: 1 = En Riesgo, 0 = No en Riesgo
        # Usando la asignación correcta: Withdrawn=0, Fail=1
        df_abt['is_at_risk'] = df_abt['final_result'].apply(lambda x: 1 if x == 1 else 0)
        
        target_column = 'is_at_risk'
        # Excluir identificadores y la variable objetivo original y categórica
        feature_columns = [col for col in df_abt.columns if col not in [
            'student_id', 'course_id', 'presentation_id', 'final_result', 'is_at_risk'
        ]]
        print(df_abt['final_result'].value_counts(dropna=False)) # dr
        # Limpieza final de datos antes de entrenar
        df_abt.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_abt.fillna(0, inplace=True)
        print(df_abt['final_result'].value_counts(dropna=False))
        
        # 3. DEFINIR Y EJECUTAR LOS MODELOS
        models_to_run = [
            ('Regresion_Logistica', LogisticRegression(max_iter=1000, random_state=42)),
            ('Random_Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('LightGBM', lgb.LGBMClassifier(random_state=42))
        ]

        all_metrics_summary = []

        for model_name, model_instance in models_to_run:
            print("\n" + "="*80)
            print(f"PROCESANDO MODELO: {model_name}")
            print("="*80)
            
            model_wrapper = StudentOutcomeModel(model=model_instance)
            
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = model_wrapper.train(
                df_abt, feature_cols=feature_columns, target_col=target_column
            )
            
            model_wrapper.evaluate_and_plot(X_test_scaled, y_test)
            
            y_pred = model_wrapper.trained_model.predict(X_test_scaled)
            
            predictions_df = model_wrapper.get_predictions_df(X_test, y_test, y_pred)
            case_by_case_filename = f'predictions_case_by_case_{model_name}.csv'
            predictions_df.to_csv(case_by_case_filename, index=False)
            print(f"Guardado el reporte caso a caso en: {case_by_case_filename}")
            
            summary = model_wrapper.get_summary_metrics(y_test, y_pred)
            all_metrics_summary.append(summary)
            
            model_wrapper.plot_feature_importance(X_train.columns)

        # 4. CREAR Y GUARDAR EL REPORTE GENERAL DE MODELOS
        summary_df = pd.DataFrame(all_metrics_summary)
        summary_filename = 'general_model_metrics.csv'
        summary_df.to_csv(summary_filename, index=False)
        print("\n" + "="*80)
        print(f"Guardado el reporte general de métricas en: {summary_filename}")
        print(summary_df)
        print("¡Proceso de modelado y exportación completado!")