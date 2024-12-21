import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

# Create an app object using Flask
app = Flask(__name__)

# Load the trained model (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form submission
    int_features = [float(x) for x in request.form.values()]  # Convert string inputs to float.
    feature_names = ['ram_gb', 'ssd', 'graphic_card_gb']      # Features used in the model.
    features = pd.DataFrame([int_features], columns=feature_names)  # Create a DataFrame.

    # Predict the price using the model
    prediction = model.predict(features)  # Features must match the training input format.
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Price of the Laptop: ₹{output}')

# Run the application
if __name__ == "__main__":
   app.run(debug=True, port=5001)
