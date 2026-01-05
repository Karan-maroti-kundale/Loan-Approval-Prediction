from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# ===============================
# Auto-train model if not present
# ===============================
if not os.path.exists("model/loan_model.pkl"):
    os.system("python model_training.py")

# ===============================
# Load Model Artifacts
# ===============================
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_names = pickle.load(open("model/features.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ===============================
        # Collect User Input
        # ===============================
        input_data = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Credit_History": float(request.form["Credit_History"]),
            "Property_Area": request.form["Property_Area"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode input
        input_df = pd.get_dummies(input_df)

        # Add missing columns from training
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure correct column order
        input_df = input_df[feature_names]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error occurred. Please check input values."
        )

# ===============================
# Run App (Render Compatible)
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
