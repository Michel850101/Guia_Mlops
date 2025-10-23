# Makefile

install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

train:
	python3 src/train.py

validate:
	python3 src/validate.py

# Ejecuta el servicio FastAPI para pruebas locales
serve:
	uvicorn src.serve:app --host 127.0.0.1 --port 8000 --workers 1

# Prueba automática del endpoint /health dentro del pipeline
serve-smoke:
	uvicorn src.serve:app --host 127.0.0.1 --port 8000 --workers 1 & echo $$! > uvicorn.pid
	sleep 3
	curl -sf http://127.0.0.1:8000/health
	curl -sf http://127.0.0.1:8000/info
	printf '{ "data": [[0.03,0.05,0.06,0.02,-0.04,-0.03,-0.04,-0.002,0.019,-0.017]] }' > body.json
	curl -sf -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @body.json
	-kill `cat uvicorn.pid` 2>/dev/null || true
	-wait `cat uvicorn.pid` 2>/dev/null || true
	-rm -f uvicorn.pid body.json