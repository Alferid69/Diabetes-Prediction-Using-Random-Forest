import pickle
from flask import Flask, request, render_template
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
    prediction = model.predict(input_array)[0]
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

    # Define prediction message and explanation
    if prediction == 1:
        prediction_message = "Diabetes Positive"
        prediction_explanation = "The model predicts that the patient is likely to have diabetes."
    else:
        prediction_message = "Diabetes Negative"
        prediction_explanation = "The model predicts that the patient is not likely to have diabetes."

    # Render the result template with prediction and explanation
    return render_template('result.html', prediction_message=prediction_message, prediction_explanation=prediction_explanation)

if __name__ == '__main__':
    app.run(debug=True)
