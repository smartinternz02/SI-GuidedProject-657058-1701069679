import numpy as np
import pickle
import pandas as pd
import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')
# scale = pickle.load(open(r'C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/rainfall_prediction/IBM flask push/Rainfall IBM deploy/scale.pkl', 'rb'))

@app.route('/')  # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web UI
def result():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['gender', 'age', 'category', 'quantity', 'price','payment_method', 'shopping_mall']
    data = pd.DataFrame(features_values, columns=names)

    # predictions using the loaded model file
    prediction = model.predict(data)
    prediction = prediction.astype(int)
    print(prediction)
    
    prediction_text = ""
    if prediction == 0:
        prediction_text = "Customer belongs to cluster label 1."
    elif prediction == 1:
        prediction_text = "Customer belongs to cluster label 2."
    else:
        prediction_text = "Highly belongs to cluster label 3 customer"
        
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
