### Experiment Tracking Lab
#### Credit Risk Predictor 
This lab builds a machine learning model to predict whether a person is a "Good" or "Bad" credit risk, based on the German Credit Data.

The main goal is to show how to use tools like MLflow to track our experiments, find the best model, and save it for later use.

How It Works
- Get Data Ready: We load the credit data and clean it up by filling in missing values and turning text (like 'male'/'female') into numbers.
- Create a "Risk" Label: The original data doesn't have a simple "good" or "bad" risk label. We use a clustering algorithm (KMeans) to automatically group all the customers into two categories, which we then call our "Risk" label.
- Train a Basic Model: We first train a simple RandomForest model. We save its accuracy (AUC score) to MLflow as our "baseline" score to beat.
- Find a Better Model: We use a powerful model (XGBoost) and a smart tuning tool (Hyperopt) to automatically test 30 different combinations of settings. This helps us find the best possible version of the model.
- Save the Winner: As the tuning runs, MLflow keeps track of every experiment. When it's finished, we ask MLflow for the model with the highest score and save it as the final, "Production" version.

Tools Used
Pandas: For loading and cleaning data.
Scikit-learn: For data scaling, clustering (KMeans), and the basic model (RandomForest).
XGBoost: For the advanced model.
Hyperopt: For automatically tuning the model's settings.
MLflow: For tracking all our experiment scores and saving the best model.
PySpark: To help Hyperopt run its tests faster in parallel.

How to Run
Install the Tools:

Bash
pip install pandas scikit-learn xgboost hyperopt mlflow pyspark
Start the MLflow Tracker: Open your terminal and run this. It starts a website where you can see your results.

Bash
mlflow ui
(You can view this at http://localhost:5000)

Run the Notebook: Open and run the lab.ipynb notebook from top to bottom.

Check Your Results: Go back to the MLflow website (http://localhost:5000). You will see all your experiments, their scores, and the final model saved under the "Models" tab.