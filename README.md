
# TC5061 - Experimentación con MLflow

## Alumno:

Jesse Rey Alvarado Valdez

## Descripción:

Se utiliza el conjunto de datos Bank Marketing de UCI para predecir si un cliente contratará un depósito bancario, utilizando regresión logística.

El objetivo es comparar diferentes configuraciones de hiperparámetros y registrar los resultados de manera reproducible mediante MLflow Tracking.

## Usaremos:

- Python 3.14.4
- Scikit-learn
- Pandas y NumPy
- Matplotlib
- MLflow Tracking

## Archivos principales:

- `train.py`: carga de datos, preprocesamiento, entrenamiento, evaluación y registro en MLflow.
- `requirements.txt`: dependencias y versiones utilizadas.
- `bank-full.csv`: conjunto de datos Bank Marketing.

## Instalación:

Desde Ubuntu o WSL, ubicarse en la carpeta del proyecto y crear un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución:

Ejecutar el modelo con los hiperparámetros predeterminados:

```bash
python3 train.py
```

Ejemplo con parámetros personalizados:

```bash
python3 train.py --C 0.1 --solver lbfgs --max_iter 1000 --random_state 42
```

Los argumentos disponibles son `--C`, `--solver`, `--max_iter` y `--random_state`. Todos cuentan con valores predeterminados y pueden consultarse mediante:

```bash
python3 train.py --help
```

## MLflow tracking:

Cada ejecución registra los hiperparámetros, cinco métricas de evaluación, la matriz de confusión, la curva ROC, un archivo JSON de métricas y el pipeline entrenado.

Para iniciar la interfaz de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir `http://127.0.0.1:5000` en el navegador y seleccionar el experimento `TC5061_Bank_Marketing_Experimentos`.

## Reproducibilidad:

El conjunto de datos se divide en 80 % para entrenamiento y 20 % para prueba, utilizando estratificación.

Se utiliza la semilla `random_state=42` en la configuración predeterminada. El preprocesamiento se integra en un Pipeline para evitar ajustar las transformaciones sobre los datos de prueba.

Las versiones de las dependencias están fijadas en `requirements.txt`.

## Limitaciones al momento:

El conjunto de datos presenta desbalance entre clases y por eso además de accuracy se consideran precision, recall, F1-score y ROC-AUC.

La variable `duration` contiene información conocida después de la llamada telefónica. En resumen este modelo es un ejercicio de experimentación y no deberia interpretarse como una herramienta lista para predecir resultados antes de contactar a los clientes.

-Jesse Rey A.


