
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# Rutas y nombre del experimento
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "bank-full.csv"
TRACKING_DB = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "TC5061_Bank_Marketing_Experimentos"


def parse_arguments():
    """Define los hiperparametros configurables desde terminal."""

    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado de Bank Marketing con MLflow"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de regularizacion. Debe ser mayor que cero."
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear"],
        help="Algoritmo de optimizacion de LogisticRegression."
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Numero maximo de iteraciones del entrenamiento."
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad."
    )

    return parser.parse_args()


def load_data():
    """Carga el dataset y separa las variables predictoras."""

    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Se encontraron valores no validos en la variable y.")

    return X, y


def build_pipeline(X, args):
    """Construye el preprocesamiento y el clasificador."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    classifier = LogisticRegression(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return pipeline


def train_and_evaluate(pipeline, X, y, random_state):
    """Divide los datos, entrena y evalua el modelo."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return metrics, y_test, y_pred, y_prob


def log_evaluation_artifacts(y_test, y_pred, y_prob, metrics):
    """Genera y registra los artefactos de evaluacion."""

    with tempfile.TemporaryDirectory() as temp_dir:

        output_dir = Path(temp_dir)

        # Matriz de confusion
        fig, ax = plt.subplots(figsize=(7, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues"
        )

        ax.set_title("Matriz de confusion - Bank Marketing")
        fig.tight_layout()

        confusion_path = output_dir / "confusion_matrix.png"
        fig.savefig(confusion_path, dpi=180)
        plt.close(fig)

        # Curva ROC
        fig, ax = plt.subplots(figsize=(7, 5))

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax
        )

        ax.set_title("Curva ROC - Bank Marketing")
        fig.tight_layout()

        roc_path = output_dir / "roc_curve.png"
        fig.savefig(roc_path, dpi=180)
        plt.close(fig)

        # Archivo JSON con las metricas
        metrics_path = output_dir / "metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # Registro de artefactos en MLflow
        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="evaluation"
        )

        mlflow.log_artifact(
            str(roc_path),
            artifact_path="evaluation"
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation"
        )


def configure_mlflow():
    """Configura el almacenamiento y experimento de MLflow."""

    mlflow.set_tracking_uri(
        f"sqlite:///{TRACKING_DB}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)


def main():

    args = parse_arguments()

    # Validaciones
    if args.C <= 0:
        raise ValueError("--C debe ser mayor que cero.")

    if args.max_iter <= 0:
        raise ValueError("--max_iter debe ser mayor que cero.")

    configure_mlflow()

    X, y = load_data()
    pipeline = build_pipeline(X, args)

    print("\nProyecto TC5061 - Experimentacion con MLflow")

    print("\nConfiguracion:")
    print(f"C: {args.C}")
    print(f"Solver: {args.solver}")
    print(f"Max iteraciones: {args.max_iter}")
    print(f"Semilla: {args.random_state}")

    print(f"\nRegistros cargados: {len(X)}")
    print(f"Variables predictoras: {X.shape[1]}")

    run_name = (
        f"C{args.C}_{args.solver}_"
        f"iter{args.max_iter}_seed{args.random_state}"
    )

    # Inicio del registro del experimento
    with mlflow.start_run(run_name=run_name) as run:

        # Registrar parametros
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("model", "LogisticRegression")

        print("\nEntrenando modelo...")

        metrics, y_test, y_pred, y_prob = train_and_evaluate(
            pipeline,
            X,
            y,
            args.random_state
        )

        # Registrar metricas
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # Registrar graficas y archivo de metricas
        log_evaluation_artifacts(
            y_test,
            y_pred,
            y_prob,
            metrics
        )

        # Registrar el pipeline completo como modelo MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="bank_marketing_model"
        )

        print("\nResultados del modelo:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experimento: {EXPERIMENT_NAME}")
        print("\nExperimento registrado correctamente en MLflow.")


if __name__ == "__main__":
    main()

