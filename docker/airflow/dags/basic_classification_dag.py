from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dag = DAG(
    'basic_classification_dag',
    default_args={'start_date': days_ago(1)},
    schedule_interval='0 23 * * *',
    catchup=False
)

def import_data():
    # Paso 1: Cargar el archivo CSV
    ruta_archivo = 'https://raw.githubusercontent.com/armandoordonez/eda_couse/main/data/prediccion_prestamo_train.csv'  # Reemplaza con la ruta de tu archivo CSV
    data = pd.read_csv(ruta_archivo)

    data = data.dropna()
    print(data.info())

    # Paso 2: Seleccionar las columnas de interés
    X = data[[" ingresos_solicitante", " ingresos_cosolicitante", " monto_prestamo"]]
    Y = data[" estado_prestamo"]

    Y = Y.replace({'Y': 1, 'N': 0})
    
    X = X.to_json(orient="records", lines=True, index=False)
    Y = Y.to_json(orient="records", lines=True, index=False)

    Variable.set("X", X)
    Variable.set("Y", Y)

def standardization():
    X = pd.read_json(Variable.get("X"), orient="records", lines=True)
    Y = pd.read_json(Variable.get("Y"), orient="records", lines=True)

    print(X.head(1))
    print(Y.head(1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.to_json(orient="records", lines=True, index=False)
    Y_train = Y_train.to_json(orient="records", lines=True, index=False)

    X_test = X_test.to_json(orient="records", lines=True, index=False)
    Y_test = Y_test.to_json(orient="records", lines=True, index=False)

    Variable.set("X_train", X_train)
    Variable.set("Y_train", Y_train)

    Variable.set("X_test", X_test)
    Variable.set("Y_test", Y_test)

def training():
    X_train = pd.read_json(Variable.get("X_train"), orient="records", lines=True)
    Y_train = pd.read_json(Variable.get("Y_train"), orient="records", lines=True)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, Y_train)

    Variable.set("modelo", modelo)

    X_test = pd.read_json(Variable.get("X_test"), orient="records", lines=True)
    Y_test = pd.read_json(Variable.get("Y_test"), orient="records", lines=True)

    Y_pred = modelo.predict(X_test)

    precision = accuracy_score(Y_test, Y_pred)
    print(f"\nPrecisión del modelo: {precision:.2f}")


import_data_task = PythonOperator(
    task_id='import_data',
    python_callable=import_data,
    dag=dag
)

standardization_task = PythonOperator(
    task_id='standardization',
    python_callable=standardization,
    dag=dag
)

training_task = PythonOperator(
    task_id='training',
    python_callable=training,
    dag=dag
)

import_data_task >> standardization_task >> training_task