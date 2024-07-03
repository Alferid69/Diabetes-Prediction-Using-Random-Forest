from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('RandomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to predict outcomes
def predict_outcome(input_features):
    # Convert input to numpy array
    input_array = np.array(input_features).reshape(1, -1)
    # Perform prediction
    prediction = model.predict(input_array)
    return prediction

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])
    age = float(request.form['age'])

    # Make prediction
    input_features = [pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, diabetes_pedigree, age]
    prediction = predict_outcome(input_features)

    # Render the result template with prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
