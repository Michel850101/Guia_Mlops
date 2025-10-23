import os, mlflow, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, RedirectResponse

# ---------------------------
# Configuración base
# ---------------------------
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "diabetes-regression")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_URI_ENV = os.getenv("MODEL_URI")
RUN_ID_ENV = os.getenv("RUN_ID")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def resolve_model_uri():
    """Determina qué modelo cargar."""
    if MODEL_URI_ENV:
        return MODEL_URI_ENV, None
    if RUN_ID_ENV:
        return f"runs:/{RUN_ID_ENV}/model", RUN_ID_ENV

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(
            f"No se encontró el experimento '{EXPERIMENT}' en '{MLFLOW_TRACKING_URI}'. "
            "Ejecute 'python src/train.py' o copie la carpeta 'mlruns/'."
        )
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise RuntimeError("El experimento no tiene ejecuciones. Entrene primero.")
    run_id = runs.iloc[0]["run_id"]
    return f"runs:/{run_id}/model", run_id

# ---------------------------
# Carga de modelo
# ---------------------------
try:
    MODEL_URI, RUN_ID = resolve_model_uri()
    model = mlflow.sklearn.load_model(MODEL_URI)
    _model_error = None
except Exception as e:
    model = None
    MODEL_URI = None
    RUN_ID = None
    _model_error = str(e)

# ---------------------------
# API FastAPI en español
# ---------------------------
app = FastAPI(
    title="Servicio de Predicción | Regresión Lineal (Diabetes)",
    description=(
        "API en español que permite realizar predicciones con un modelo de regresión lineal entrenado.\n\n"
        "**Endpoints disponibles:**\n"
        "- `/health`: verifica el estado del servicio.\n"
        "- `/info`: muestra información del modelo cargado.\n"
        "- `/predict`: genera predicciones a partir de 10 características numéricas por fila.\n\n"
        "Ejemplo de entrada:\n"
        "{ \"data\": [[0.03, 0.05, 0.06, 0.02, -0.04, -0.03, -0.04, -0.002, 0.019, -0.017]] }"
    ),
    version="1.0.0",
)

@app.get("/", include_in_schema=False)
def root():
    """Redirige a la documentación interactiva."""
    return RedirectResponse(url="/docs")

class Features(BaseModel):
    data: list = []  # lista de filas, cada fila con 10 valores numéricos

@app.exception_handler(Exception)
async def general_error(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "detalle": str(exc),
            "sugerencia": "Revise la carpeta 'mlruns' o defina MODEL_URI/RUN_ID antes de ejecutar."
        },
    )

@app.get("/health")
def health():
    """Verifica si el modelo está disponible."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "estado": "no_disponible",
                "mensaje": "Servicio activo, pero el modelo no pudo cargarse.",
                "detalle": _model_error,
            },
        )
    return {"estado": "ok", "mensaje": "Servicio y modelo listos para inferencia."}

@app.get("/info")
def info():
    """Devuelve información sobre el modelo cargado."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no cargado. {_model_error}")
    return {
        "mensaje": "Modelo actualmente cargado para inferencia.",
        "experimento": EXPERIMENT,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "run_id": RUN_ID,
        "model_uri": MODEL_URI,
    }

@app.post("/predict")
def predict(payload: Features):
    """Recibe una lista de filas con 10 valores cada una y devuelve las predicciones."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    try:
        df = pd.DataFrame(payload.data)
        if df.shape[1] != 10:
            raise ValueError(f"Se esperaban 10 columnas, se recibieron {df.shape[1]}.")
        preds = model.predict(df).tolist()
        return {
            "predicciones": preds,
            "explicacion": "Cada valor corresponde a una predicción del modelo para la fila enviada."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No fue posible realizar la predicción: {e}")