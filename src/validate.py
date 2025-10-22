"""
Script de validación del modelo
Valida que el MSE del último run sea aceptable antes de promover el modelo
"""
import os
import sys
import mlflow
from pathlib import Path


def setup_tracking_uri():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
os.environ.pop("MLFLOW_ARTIFACT_URI", None)
threshold_mse = float(os.getenv("THRESHOLD_MSE", "3000"))

def get_latest_run(experiment_name, threshold_mse=3000):
    """
    Obtiene el último run del experimento y valida el MSE
    
    Args:
        experiment_name: Nombre del experimento
        threshold_mse: Umbral máximo aceptable de MSE
    
    Returns:
        run_id del modelo validado o None si falla
    """
    print(f"\n{'='*60}")
    print(f"Validando modelo del experimento: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Obtener experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"❌ Experimento '{experiment_name}' no encontrado")
        return None
    
    print(f"✓ Experimento encontrado (ID: {experiment.experiment_id})")
    
    # Buscar runs ordenados por fecha
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        print("❌ No se encontraron runs en el experimento")
        return None
    
    # Obtener métricas del último run
    latest_run = runs.iloc[0]
    run_id = latest_run["run_id"]
    mse = latest_run["metrics.test_mse"]
    mae = latest_run.get("metrics.test_mae", "N/A")
    r2 = latest_run.get("metrics.test_r2", "N/A")
    
    print(f"\n📊 Métricas del último run:")
    print(f"   Run ID: {run_id}")
    print(f"   Test MSE: {mse:.4f}")
    if mae != "N/A":
        print(f"   Test MAE: {mae:.4f}")
    if r2 != "N/A":
        print(f"   Test R²:  {r2:.4f}")
    
    # Validar MSE
    print(f"\n🔍 Validando MSE contra umbral: {threshold_mse}")
    
    if mse > threshold_mse:
        print(f"\n❌ MSE demasiado alto ({mse:.4f} > {threshold_mse})")
        print("   El modelo NO pasa la validación")
        print("   No se puede promover a producción")
        return None
    else:
        print(f"\n✅ MSE aceptable ({mse:.4f} <= {threshold_mse})")
        print("   El modelo PASA la validación")
        print("   Listo para promoción a producción")
        return run_id


def promote_model(run_id, model_name="diabetes-linear-regression"):
    """
    Promueve el modelo a la etapa 'Production' en el Model Registry
    
    Args:
        run_id: ID del run a promover
        model_name: Nombre del modelo registrado
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Buscar la versión del modelo asociada al run_id
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        target_version = None
        for mv in model_versions:
            if mv.run_id == run_id:
                target_version = mv.version
                break
        
        if target_version:
            # Transicionar a Production
            client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )
            print(f"\n🚀 Modelo '{model_name}' versión {target_version} promovido a Production")
            return True
        else:
            print(f"\n⚠️  No se encontró versión del modelo para el run {run_id}")
            return False
            
    except Exception as e:
        print(f"\n⚠️  Error al promover modelo: {str(e)}")
        print("   Continuando sin promoción automática...")
        return False


def main():
    """Función principal"""
    try:
        # Configurar MLflow
        setup_tracking_uri()
        
        # Nombre del experimento (debe coincidir con train.py)
        experiment_name = "diabetes-regression"
        
        # Umbral de MSE (ajústalo según necesites)
        threshold_mse = float(os.getenv("THRESHOLD_MSE", "3000"))
        
        # Validar último run
        run_id = get_latest_run(experiment_name, threshold_mse)
        
        if run_id is None:
            print(f"\n{'='*60}")
            print("RESULTADO: VALIDACIÓN FALLIDA")
            print(f"{'='*60}\n")
            return 1
        
        # Intentar promover el modelo
        promote_model(run_id)
        
        print(f"\n{'='*60}")
        print("RESULTADO: VALIDACIÓN EXITOSA")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error durante la validación: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())