# Makefile para automatizar el Protocolo LAPA-CR (Fase de Evaluación)
# Nota: La generación de voces (XTTS) se mantiene manual por requerir otro entorno.

.PHONY: all identidad frecuencia calidad limpiar

# Comando global que corre todo el pipeline de evaluación
all: identidad frecuencia calidad
	@echo "\n========================================================"
	@echo "   Protocolo LAPA-CR: Evaluación Completa Finalizada    "
	@echo "========================================================\n"

# 1. Evaluación Biométrica (ECAPA-TDNN)
identidad:
	@echo "\n---> Iniciando Módulo de Identidad (ECAPA-TDNN) <---"
	python scripts/evaluacion/calcular_eer_general.py
	python scripts/evaluacion/visualizar_resultados.py
	@mv *.png outputs/graficas/ 2>/dev/null || true

# 2. Evaluación Frecuencial (MCD y Espectrogramas)
frecuencia:
	@echo "\n---> Iniciando Módulo de Distorsión Espectral (MCD) <---"
	python scripts/evaluacion/calcular_mcd.py
	@echo "\n---> Generando Mapas de Calor de Frecuencias <---"
	python scripts/evaluacion/graficar_espectrogramas.py

# 3. Evaluación de Calidad (NISQA)
calidad:
	@echo "\n---> Iniciando Módulo de Calidad Perceptual (NISQA) <---"
	python scripts/evaluacion/calcular_mos_nisqa.py
	@mv resultados_nisqa/* outputs/datos_crudos/ 2>/dev/null || true

# Comando de limpieza
limpiar:
	@echo "Limpiando archivos temporales y cachés..."
	rm -rf __pycache__ .pytest_cache
	rm -rf scripts/evaluacion/__pycache__
	@echo "Limpieza completada."
