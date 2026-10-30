from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained KNN model
model = joblib.load("KNNDiabetesModel.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get input values from form
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        diabetes_pedigree = float(
            request.form["DiabetesPedigreeFunction"]
        )
        age = float(request.form["Age"])

        # Create input data in the same order as training data
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 1:
            result = "Diabetes Detected"
        else:
            result = "No Diabetes"

        return render_template(
            "index.html",
            prediction=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error: " + str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)

    
import pickle

# Save the trained KNN model
with open("KNNDiabetesModel.pkl", "wb") as file:
    pickle.dump(model, file)

print("KNN Diabetes model saved successfully!")
