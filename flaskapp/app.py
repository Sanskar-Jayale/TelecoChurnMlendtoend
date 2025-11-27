#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import joblib
from flask import Flask, request, session, render_template, flash

app = Flask(__name__)

app.config.update(dict(
    DEBUG=True,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'development key')
))

# Fields
strings = {
    "gender": ['Female', 'Male'],
    "Partner": ['Yes', 'No'],
    "Dependents": ['No', 'Yes'],
    "PhoneService": ['No', 'Yes'],
    "MultipleLines": ['No phone service', 'No', 'Yes'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
    "OnlineSecurity": ['No', 'Yes', 'No internet service'],
    "OnlineBackup": ['Yes', 'No', 'No internet service'],
    "DeviceProtection": ['No', 'Yes', 'No internet service'],
    "TechSupport": ['No', 'Yes', 'No internet service'],
    "StreamingTV": ['No', 'Yes', 'No internet service'],
    "StreamingMovies": ['No', 'Yes', 'No internet service'],
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "PaperlessBilling": ['Yes', 'No'],
    "PaymentMethod": ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

floats = {
    "MonthlyCharges": [0, 1000, 100],
    "TotalCharges": [0, 50000, 1000]
}

ints = {
    "SeniorCitizen": [0, 1, 0],
    "tenure": [0, 100, 2]
}

labels = ["No Churn", "Churn"]

# Load trained pipeline
MODEL_PATH = r'D:\Sanskar\TelecoChurnMlendtoend\Notebook\churn_random_forest.pkl'
clf = joblib.load(MODEL_PATH)

# Function to generate form HTML
def generate_input_lines():
    result = f'<table>'
    counter = 0

    for k in floats.keys():
        minn, maxx, vall = floats[k]
        if counter % 2 == 0: result += '<tr>'
        result += f'<td>{k}<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" value="{vall}" required></td>'
        if counter % 2 == 1: result += '</tr>'
        counter += 1

    counter = 0
    for k in ints.keys():
        minn, maxx, vall = ints[k]
        if counter % 2 == 0: result += '<tr>'
        result += f'<td>{k}<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" value="{vall}" required></td>'
        if counter % 2 == 1: result += '</tr>'
        counter += 1

    counter = 0
    for k in strings.keys():
        if counter % 2 == 0: result += '<tr>'
        result += f'<td>{k}<select class="form-control" name="{k}">'
        for value in strings[k]:
            result += f'<option value="{value}" selected>{value}</option>'
        result += '</select></td>'
        if counter % 2 == 1: result += '</tr>'
        counter += 1

    result += '</table>'
    return result

app.jinja_env.globals.update(generate_input_lines=generate_input_lines)

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data
        input_data = {}
        for k, v in request.form.items():
            if k in ints:
                input_data[k] = int(v)
            elif k in floats:
                input_data[k] = float(v)
            else:
                input_data[k] = v

        input_df = pd.DataFrame([input_data])

        # Predict
        pred = clf.predict(input_df)[0]
        prob = clf.predict_proba(input_df)[0]

        churn_risk = 'Churn' if pred == 1 else 'No Churn'
        yes_percent = prob[1] * 100
        no_percent = prob[0] * 100

        flash(f'Percentage of this customer leaving is: {yes_percent:.0f}%')

        # Create result dictionary for JS
        result = {
            "predictedLabel": churn_risk,
            "probability": prob.tolist()  # convert numpy array to list
        }

        return render_template(
            'score.html',
            result=result,
            churn_risk=churn_risk,
            yes_percent=yes_percent,
            no_percent=no_percent,
            labels=labels
        )

    else:
        return render_template('input.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
