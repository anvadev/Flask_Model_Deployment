# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_iris.pkl')

@app.route('/')
def home():
    return render_template('Flask_UI.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form input data
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        # Standardize data
        input_data = np.array([[sl, sw, pl, pw]])
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)

        # Predictions using model_iris.pkl
        prediction = model.predict(input_data)

        # Map numeric predictions to class names
        species = {
            0: 'Setosa',
            1: 'Versicolor',
            2: 'Virginica'
        }

        # Predicted species name
        predicted_species = species[prediction[0]]

        return render_template('Flask_UI.html', prediction=f'Predicted Species: {predicted_species}')

    except Exception as e:
        return render_template('Flask_UI.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
