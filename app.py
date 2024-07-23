from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the pre-trained machine learning model and label encoder
with open(r"C:\Users\annab\Downloads\chatbot\model (2).pkl", 'rb') as file:
    model = pickle.load(file, encoding='latin1')
label_encoder = pickle.load(open("label_encoder.joblib", "rb"))

# Create a StandardScaler instance
scaler = StandardScaler()

questions = [
    "What is your gender? (Male or Female)",
    "What is your age?",
    "Do you have hypertension? (1 for Yes, 0 for No)",
    "Do you have heart disease? (1 for Yes, 0 for No)",
    "Are you ever married? (Yes or No)",
    "What is your work type? (e.g., Private, Self-employed, Govt_job)",
    "What is your residence type? (Urban or Rural)",
    "What is your average glucose level?",
    "What is your BMI?",
    "Do you smoke? (Smokes, Formerly smoked, Never smoked)"
]

@app.route('/')
def index():
    return render_template('index.html', questions=questions)
@app.route('/')
def welcome():
    return render_template('index.html')

# Route for the chatbot page
@app.route('/chat')
def chat():
    return render_template('chat.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = request.form['ever_married']
    work_type = request.form['work_type']
    residence_type = request.form['residence_type']
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['smoking_status']

    # Preprocess the input data
    scaled_numerical_data = preprocess_numerical(age, hypertension, heart_disease, avg_glucose_level, bmi)
    categorical_data = preprocess_categorical(gender, ever_married, work_type, residence_type, smoking_status)

    # Make prediction
    prediction = model.predict([scaled_numerical_data, categorical_data.reshape(1, -1)])

    # Convert prediction to human-readable format
    result = "likely to have a stroke" if prediction[0] >= 0.5 else "unlikely to have a stroke"

    return render_template('index.html', questions=questions, prediction_text='Based on the provided information, the person is {}'.format(result))

def preprocess_numerical(age, hypertension, heart_disease, avg_glucose_level, bmi):
    # Scale the input numerical data using the loaded scaler
    numerical_data = np.array([age, hypertension, heart_disease, avg_glucose_level, bmi])
    scaled_numerical_data = scaler.transform([numerical_data])

    return scaled_numerical_data

def preprocess_categorical(gender, ever_married, work_type, residence_type, smoking_status):
    # Encode categorical features
    gender_encoded = label_encoder.transform([gender])[0]
    ever_married_encoded = label_encoder.transform([ever_married])[0]
    work_type_encoded = label_encoder.transform([work_type])[0]
    residence_type_encoded = label_encoder.transform([residence_type])[0]
    smoking_status_encoded = label_encoder.transform([smoking_status])[0]

    # Create input array for categorical data in the same order as X_train_categorical
    categorical_data = np.array([gender_encoded, ever_married_encoded, work_type_encoded, residence_type_encoded, smoking_status_encoded])

    return categorical_data

if __name__ == '__main__':
    app.run(debug=True)
