Wine Classifier Lab
This lab trains and serves a wine classification model using Docker, TensorFlow, and Flask. You can use a web page or API to get predictions based on wine features.

How to Use
1. Build the Docker image:
    docker build -t wine-app .

2. Run the app:
    docker run -p 4000:4000 wine-app

3. Get predictions:
    Visit http://localhost:4000/predict in your browser and fill in the features.

    Or send a POST request with all 13 features to /predict.

Files
1. dockerfile: Instructions for building/training/serving.
2. requirements.txt: Needed Python packages.
3. src/model_training.py: Trains and saves the wine model.
4. src/main.py: Runs the Flask web/API server.
5. src/templates/predict.html: The web input form.

Notes
Enter all 13 wine features for predictions.
Everything runs inside Docker—no extra setup needed.

