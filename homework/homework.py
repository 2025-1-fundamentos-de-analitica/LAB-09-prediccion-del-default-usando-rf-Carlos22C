# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def cargar_dataset(path):
    return pd.read_csv(path, compression="zip")

def limpiar_dataset(df):
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns="ID", inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df

def separar_variables(df_train, df_test):
    x_tr = df_train.drop(columns=["default"])
    y_tr = df_train["default"]
    x_te = df_test.drop(columns=["default"])
    y_te = df_test["default"]
    return x_tr, y_tr, x_te, y_te

def construir_pipeline():
    cat_cols = ["EDUCATION", "SEX", "MARRIAGE"]
    preproc = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")

    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline(steps=[
        ("transformacion", preproc),
        ("clasificador", clf)
    ])
    return pipe

def optimizar_modelo(pipeline, X, y):
    params = {
        "clasificador__n_estimators": [300],
        "clasificador__max_depth": [30],
        "clasificador__min_samples_split": [5],
        "clasificador__min_samples_leaf": [1],
        "clasificador__max_features": ["sqrt"]
    }
    gs = GridSearchCV(pipeline, params, scoring="balanced_accuracy", cv=10, n_jobs=-1, verbose=0)
    return gs.fit(X, y)

def guardar_modelo_trenado(modelo, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with gzip.open(destino, "wb") as salida:
        pickle.dump(modelo, salida)

def obtener_metricas(modelo, X, y, tipo):
    y_hat = modelo.predict(X)
    return {
        "type": "metrics",
        "dataset": tipo,
        "precision": round(precision_score(y, y_hat), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_hat), 4),
        "recall": round(recall_score(y, y_hat), 4),
        "f1_score": round(f1_score(y, y_hat), 4)
    }, y_hat

def generar_matriz_confusion(y_real, y_pred, tipo):
    matriz = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": tipo,
        "true_0": {"predicted_0": int(matriz[0][0]), "predicted_1": int(matriz[0][1])},
        "true_1": {"predicted_0": int(matriz[1][0]), "predicted_1": int(matriz[1][1])}
    }

def main():
    os.makedirs("files/output", exist_ok=True)
    train = limpiar_dataset(cargar_dataset("files/input/train_data.csv.zip"))
    test = limpiar_dataset(cargar_dataset("files/input/test_data.csv.zip"))

    X_train, y_train, X_test, y_test = separar_variables(train, test)
    pipeline = construir_pipeline()
    mejor = optimizar_modelo(pipeline, X_train, y_train)

    mtrain, pred_train = obtener_metricas(mejor, X_train, y_train, "train")
    mtest, pred_test = obtener_metricas(mejor, X_test, y_test, "test")

    ctrain = generar_matriz_confusion(y_train, pred_train, "train")
    ctest = generar_matriz_confusion(y_test, pred_test, "test")

    with open("files/output/metrics.json", "w") as f:
        for fila in [mtrain, mtest, ctrain, ctest]:
            f.write(json.dumps(fila) + "\n")

    guardar_modelo_trenado(mejor, "files/models/model.pkl.gz")

if __name__ == "__main__":
    main()
