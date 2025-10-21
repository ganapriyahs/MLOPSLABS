Project Overview & Achievement I built a machine learning pipeline using Apache Airflow and Docker. The setup is clean, modular, and easy to reuse or change for other projects.

Key Components

Airflow Workflow (diabetes_pipeline.py): 
- Preprocessing: Cleans the original diabetes data using preprocess.py.
- Training: Trains two models (model_logreg.pkl and model_rf.pkl) using train_model.py.
- Evaluation: Checks model performance and writes a report with evaluate.py.
- Testing: Runs tests and creates a log with test.py.

Project Folders: 
- assets: Extra files or images for support
- config: Files to control pipeline settings dags: The Airflow workflow scripts
- data: Raw and cleaned dataset files
- output: Saved models, reports, and logs
- scripts: Python code for each pipeline step

Setup and Tools: 
- docker-compose.yaml: Makes it easy to start Airflow using Docker
- requirements.txt: Lists all the Python packages needed
- setup.sh: Script to help set up everything quickly
- .gitignore, .DS_Store: Keeps version control tidy
