<h1 align="center">🎓 Maestría en Ciencia de Datos</h1>
<h2 align="center">MLOps y Analítica en la Nube</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-FastAPI-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tracking-MLflow-orange.svg" alt="MLflow">
  <img src="https://img.shields.io/badge/CI/CD-GitHub_Actions-black.svg" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

---

### 👤 Autor
**Michel Stivens Larrota Villalba**

### 🧾 Proyecto
**Examen Final - Implementación de Pipeline MLOps con MLflow y FastAPI**

---

### 🧠 Descripción
Repositorio académico correspondiente al curso **MLOps y Analítica en la Nube**, de la **Maestría en Ciencia de Datos**.  
Incluye la configuración, entrenamiento, validación y despliegue de un modelo de Machine Learning mediante **MLflow**, **FastAPI** y **GitHub Actions** con enfoque CI/CD.

---

### 🗂️ Estructura del proyecto
---

### ⚙️ Tecnologías y librerías
- **Python 3.10+**
- **MLflow** – tracking y registro de modelos  
- **FastAPI** – servicio de inferencia REST  
- **Scikit-learn** – modelo base  
- **GitHub Actions** – CI/CD automatizado  
- **Makefile** – tareas reproducibles  
- **Azure ML / Local environment** – despliegue flexible  

---

### 🚀 Ejecución rápida
```bash
# Clonar el repositorio
git clone https://github.com/Michel850101/mlflow-deploy.git
cd mlflow-deploy

# Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate    # Windows
# o
source .venv/bin/activate        # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
python src/train.py

# Validar modelo
python src/validate.py

# Servir API
python src/serve.py
