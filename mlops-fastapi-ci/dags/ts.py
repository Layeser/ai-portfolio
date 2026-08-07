from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract():
    print("extracting data")

def transform():
    print("transforming data")

def train():
    print("training model")

with DAG("ml_pipeline", start_date=datetime(2024,1,1), schedule_interval="*/5 * * * *") as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="train", python_callable=train)

    t1 >> t2 >> t3