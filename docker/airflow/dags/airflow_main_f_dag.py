from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

import datetime

dag = DAG(
    "airflow_main_f_dag",
    default_args={"start_date": days_ago(1)},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=120)
)

def import_data():
    warnings.filterwarnings("ignore")

    icfes_path = "https://raw.githubusercontent.com/kelocoes/EDA-Notebooks/main/datasets/SABERTYT20162.csv"
    df = pd.read_csv(icfes_path, sep=";", encoding="latin1")
    
    df = df.to_json(orient="records", lines=True, index=False)
    Variable.set("df", df)

def clean_data():
    df = Variable.get("df")
    df = pd.read_json(df, orient="records", lines=True)

    df = df.loc[~df["ESTU_ESTADO"].str.contains("VALIDEZ OFICINA JURÍDICA")] # Solo trabajar con pruebas ya validas
    df = df.loc[df["ESTU_NACIONALIDAD"].str.contains("1 COLOMBIA")] # Solo trabajar con estudiantes colombianos
    df.drop(["ESTU_ESTADO", "ESTU_NACIONALIDAD"], axis=1, inplace=True) # Eliminar columnas que no aportan información

    columns_of_cod_to_drop = df.filter(regex="COD|DESEM|PNAL|PGREF|PRESENTACION", axis=1)
    df.drop(columns=columns_of_cod_to_drop.columns, inplace=True)

    columns_to_drop = ["ESTU_TIPODOCUMENTO", "PERIODO", "ESTU_CONSECUTIVO", "ESTU_ESTUDIANTE",
                    "ESTU_ETNIA", "ESTU_COLE_TERMINO", "ESTU_VALOR_MATRICULAUNIVER", "ESTU_PAGO_MATRICULA_PADRES", 
                    "ESTU_PAGO_MATRICULA_CREDITO", "ESTU_PAGO_MATRICULA_PROPIO", "ESTU_PAGO_MATRICULA_BECA",
                    "ESTU_CURSO_DOCENTESIES", "ESTU_CURSO_IES_APOYOEXTERNO", "ESTU_CURSO_IESEXTERNA",
                    "ESTU_ACTIVIDAD_REFUERZOAREA", "ESTU_ACTI_REFUERZOGENERICAS", "ESTU_COMO_CAPACITOEXAMEN",
                    "ESTU_SEMESTRE_CURSA", "ESTU_TIPO_REMUNERACION", "ESTU_SIMULACRO_TIPOICFES",
                    "ESTU_OTRO_PREGRADO", "ESTU_UN_POSTGRADO", "ESTU_PREGRADO_EXAM_SBPRO",
                    "ESTU_CURSO_NOPREGRADO", "ESTU_OTROCOLE_TERMINO", "ESTU_SNIES_PRGMACADEMICO",
                    "ESTU_ACTIVIDAD_REFUERZOAREA", "ESTU_ACTI_REFUERZOGENERICAS", "NSE", "INSE"]

    df.drop(columns=columns_to_drop, inplace=True)

    df["ESTU_FECHANACIMIENTO"] = pd.to_datetime(df["ESTU_FECHANACIMIENTO"], format="%d/%m/%Y")
    rows_to_remove = df.loc[df["ESTU_FECHANACIMIENTO"] > "2004-01-01"].index
    df.drop(rows_to_remove, inplace=True)
    df_actual_date = pd.DataFrame(columns=['actual_date'])
    df_actual_date['actual_date'] = [pd.to_datetime('2016-10-09')] * df.shape[0]
    df["EDAD"] = df_actual_date['actual_date'].sub(df['ESTU_FECHANACIMIENTO']).dt.days // 365
    df.drop("ESTU_FECHANACIMIENTO", axis=1, inplace=True)
    print(df.info())

    df = df.to_json(orient="records", lines=True, index=False)
    Variable.set("df", df)

def process_null_data():
    df = Variable.get("df")
    df = pd.read_json(df, orient="records", lines=True)

    columns_with_na = df.isna().sum().sort_values(ascending=False)
    print(f"Cantidad de columnas con NaN: {len(columns_with_na[columns_with_na > 0])}")

    df["ESTU_TIENE_ETNIA"].fillna("NO", inplace=True)
    print(f"Campos vacios ESTU_TIENE ETNIA: {df['ESTU_TIENE_ETNIA'].isna().sum()}")

    columns_dissabilities = list(df.filter(regex="LIMITA", axis=1).columns)
    for col in columns_dissabilities:
        df[col].replace(np.nan, False, inplace=True)
        df[col].replace("x", True, inplace=True)
    print(df.filter(regex="LIMITA", axis=1).isna().sum().sort_values(ascending=False))

    columns_with_na = df.isna().sum().sort_values(ascending=False)
    columns_with_na = columns_with_na[columns_with_na > 0].index.tolist()
    print(f"Columnas que faltan por rellenar {len(columns_with_na)}")

    df[columns_with_na] = df[columns_with_na].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

    columns_with_na = df.isna().sum().sort_values(ascending=False)
    columns_with_na = columns_with_na[columns_with_na > 0].index.tolist()
    print(f"Columnas que faltan por rellenar {len(columns_with_na)}")

    df = df.to_json(orient="records", lines=True, index=False)
    Variable.set("df", df)

def process_cat_data():
    df = Variable.get("df")
    df = pd.read_json(df, orient="records", lines=True)

    df["FAMI_NUM_PERSONASACARGO"].unique()
    df["FAMI_NUM_PERSONASACARGO"] = df["FAMI_NUM_PERSONASACARGO"].replace("12 O MÁS", 12).astype(int)

    cat_attribs = df.select_dtypes(include="object").columns.tolist()

    label_encoder = LabelEncoder()
    mapping_info = []

    for col in cat_attribs:
        df[col] = label_encoder.fit_transform(df[col])

        original_classes = label_encoder.classes_
        col_mapping = [(original, encoded) for original, encoded in zip(original_classes, range(len(original_classes)))]
        mapping_info.append({"Columna": col, "Mapeo": col_mapping})

    df = df.to_json(orient="records", lines=True, index=False)
    Variable.set("df", df)

def process_num_data():
    df = Variable.get("df")
    df = pd.read_json(df, orient="records", lines=True)

    float_columns = df.select_dtypes(include="float64").columns.tolist()
    df[float_columns] = df[float_columns].astype(int)

    int_32_columns = df.select_dtypes(include="int32").columns.tolist()
    df[int_32_columns] = df[int_32_columns].astype("int64")

    bool_columns = df.select_dtypes(include="bool").columns.tolist()
    df[bool_columns] = df[bool_columns].astype("int64")

    df = df.to_json(orient="records", lines=True, index=False)
    Variable.set("df", df)

def split_data():
    df = Variable.get("df")
    df = pd.read_json(df, orient="records", lines=True)

    columns_areas = ["MOD_RAZONA_CUANTITAT_PUNT", "MOD_LECTURA_CRITICA_PUNT", "MOD_COMPETEN_CIUDADA_PUNT", "MOD_INGLES_PUNT"]

    X = df.drop(columns_areas, axis=1)
    Y = df[columns_areas]

    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.to_json(orient="records", lines=True, index=False)
    x_test = x_test.to_json(orient="records", lines=True, index=False)
    Y_train = Y_train.to_json(orient="records", lines=True, index=False)
    y_test = y_test.to_json(orient="records", lines=True, index=False)
    Variable.set("X_train", X_train)
    Variable.set("x_test", x_test)
    Variable.set("Y_train", Y_train)
    Variable.set("y_test", y_test)

def train_model():
    X_train = Variable.get("X_train")
    Y_train = Variable.get("Y_train")
    X_train = pd.read_json(X_train, orient="records", lines=True)
    Y_train = pd.read_json(Y_train, orient="records", lines=True)

    x_test = Variable.get("x_test")
    y_test = Variable.get("y_test")
    x_test = pd.read_json(x_test, orient="records", lines=True)
    y_test = pd.read_json(y_test, orient="records", lines=True)

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, Y_train)

    y_pred = forest_reg.predict(x_test)

    y_test = np.asarray(y_test)

    mae_por_materia = []

    for i in range(len(y_test[0])):
        mae = mean_absolute_error([y_test[j][i] for j in range(len(y_test))], [y_pred[j][i] for j in range(len(y_pred))])
        mae_por_materia.append(mae)

    for i, mae in enumerate(mae_por_materia):
        print(f"MAE para materia {Y_train.columns[i]}: {mae}")

import_data_task = PythonOperator(
    task_id="import_data",
    python_callable=import_data,
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id="clean_data",
    python_callable=clean_data,
    dag=dag,
)

process_null_data_task = PythonOperator(
    task_id="process_null_data",
    python_callable=process_null_data,
    dag=dag,
)

process_cat_data_task = PythonOperator(
    task_id="process_cat_data",
    python_callable=process_cat_data,
    dag=dag,
)

process_num_data_task = PythonOperator(
    task_id="process_num_data",
    python_callable=process_num_data,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id="split_data",
    python_callable=split_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

import_data_task >> clean_data_task >> process_null_data_task >> process_cat_data_task >> process_num_data_task >> split_data_task >> train_model_task