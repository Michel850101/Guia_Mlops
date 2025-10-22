"""
Script de entrenamiento con MLflow
Entrena un modelo de regresión lineal con el dataset Diabetes
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models import infer_signature


def setup_mlflow():
    import mlflow
    import os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    # neutraliza valores heredados
    os.environ.pop("MLFLOW_ARTIFACT_URI", None)
    print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")

    experiment_name = "diabetes-regression"
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print(f"✓ Experimento '{experiment_name}' (ID: {exp.experiment_id})")
    return exp.experiment_id

def load_and_split_data(test_size=0.2, random_state=42):
    """Carga y divide el dataset Diabetes"""
    print("\nCargando datos...")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✓ Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


def train_and_log_model(experiment_id, X_train, X_test, y_train, y_test):
    """Entrena el modelo y registra métricas en MLflow"""
    print("\nEntrenando modelo...")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="linear_regression") as run:
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Registrar parámetros
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Registrar métricas
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        
        # Inferir signature del modelo
        signature = infer_signature(X_train, y_pred_train)
        
        # Registrar modelo en MLflow
        mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
    # quita registered_model_name para evitar Model Registry en runner efímero
)
        
        # Mostrar resultados
        print(f"\n{'='*50}")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'='*50}")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE:  {test_mse:.4f}")
        print(f"Test MAE:  {test_mae:.4f}")
        print(f"Test R²:   {test_r2:.4f}")
        print(f"{'='*50}")
        print(f"✓ Modelo registrado exitosamente")
        
        return run.info.run_id, model


def main():
    """Función principal"""
    try:
        print("Iniciando pipeline de entrenamiento...\n")
        
        # Setup MLflow
        experiment_id = setup_mlflow()
        
        # Cargar datos
        X_train, X_test, y_train, y_test = load_and_split_data()
        
        # Entrenar y registrar modelo
        run_id, model = train_and_log_model(
            experiment_id, X_train, X_test, y_train, y_test
        )
        
        print(f"\n✅ Pipeline completado exitosamente!")
        print(f"Run ID: {run_id}")
        print(f"\nPara ver resultados ejecuta: mlflow ui")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())