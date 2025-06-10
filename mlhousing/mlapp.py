from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('linear_model.pkl')  # Load the model using joblib


@app.route('/')
def index():
    return render_template('mlindex.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form values
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']

    # Convert values to NumPy array and float type
    arr = np.array([val1, val2, val3, val4], dtype=np.float64)
    
    # Make prediction
    pred = model.predict([arr])[0]  # Get the first (and only) prediction result

    # Return result to the UI
    return render_template('mlindex.html', prediction=f"${int(pred):,}")  # Format result as currency


if __name__ == '__main__':
    app.run(debug=True)
