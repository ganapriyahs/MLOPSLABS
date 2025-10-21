from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
}

with DAG('diabetes_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python3 /opt/airflow/scripts/preprocess.py /opt/airflow/data/diabetes.csv /opt/airflow/data/cleaned_diabetes.csv'
    )

    train = BashOperator(
    task_id='train',
    bash_command='python3 /opt/airflow/scripts/train_model.py /opt/airflow/data/cleaned_diabetes.csv /opt/airflow/output/model_logreg.pkl /opt/airflow/output/model_rf.pkl'
    )

    evaluate = BashOperator(
    task_id='evaluate',
    bash_command='python3 /opt/airflow/scripts/evaluate.py /opt/airflow/data/cleaned_diabetes.csv /opt/airflow/output/model_logreg.pkl /opt/airflow/output/model_rf.pkl /opt/airflow/output/report.txt'
    )

    test = BashOperator(
    task_id='test',
    bash_command='python3 /opt/airflow/scripts/test.py /opt/airflow/output/model_logreg.pkl /opt/airflow/output/test_output.log'
)



    preprocess >> train >> evaluate >> test
