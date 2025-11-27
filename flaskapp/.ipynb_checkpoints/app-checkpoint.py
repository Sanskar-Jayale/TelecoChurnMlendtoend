#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from flask import Flask, request, session, render_template, flash

app = Flask(__name__)

app.config.update(dict(
    DEBUG=True,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'development key')
))

# Options for categorical variables
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
    "PaymentMethod": ['Electronic check',
                      'Mailed check',
                      'Bank transfer (automatic)',
                      'Credit card (automatic)']
}

# min, max, default value
floats = {
    "MonthlyCharges": [0, 1000, 100],
    "TotalCharges": [0, 50000, 1000]
}

# min, max, default value
ints = {
    "SeniorCitizen": [0, 1, 0],
    "tenure": [0, 100, 2],
}

labels = ["No Churn", "Churn"]

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'D:\\Sanskar\\TelecoChurnMlendtoend\\Notebook\\churn_random_forest.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Optional: If you have a scaler
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
else:
    scaler = None


def generate_input_lines():
    result = f'<table>'
    counter = 0

    # Float fields
    for k in floats.keys():
        minn, maxx, vall = floats[k]
        if counter % 2 == 0:
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required>'
        result += f'</td>'
        if counter % 2 == 1:
            result += f'</tr>'
        counter += 1

    # Integer fields
    counter = 0
    for k in ints.keys():
        minn, maxx, vall = ints[k]
        if counter % 2 == 0:
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required>'
        result += f'</td>'
        if counter % 2 == 1:
            result += f'</tr>'
        counter += 1

    # Categorical fields
    counter = 0
    for k in strings.keys():
        if counter % 2 == 0:
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<select class="form-control" name="{k}">'
        for value in strings[k]:
            result += f'<option value="{value}" selected>{value}</option>'
        result += f'</select>'
        result += f'</td>'
        if counter % 2 == 1:
            result += f'</tr>'
        counter += 1

    result += f'</table>'
    return result


app.jinja_env.globals.update(generate_input_lines=generate_input_lines)


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        data = {}

        # Collect form data
        for k, v in request.form.items():
            data[k] = v
            session[k] = v

        # Convert types
        for field in ints.keys():
            data[field] = int(data[field])
        for field in floats.keys():
            data[field] = float(data[field])

        # Handle categorical encoding (basic example using one-hot / manual mapping)
        # For simplicity, let's assume you manually encode categories for your model
        # Replace this part with your exact preprocessing pipeline
        input_features = []
        for k in floats.keys():
            input_features.append(data[k])
        for k in ints.keys():
            input_features.append(data[k])

        # Example: convert categorical to numeric using index
        for k in strings.keys():
            input_features.append(strings[k].index(data[k]))

        input_array = np.array(input_features).reshape(1, -1)

        # Apply scaler if exists
        if scaler:
            input_array = scaler.transform(input_array)

        # Predict
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]

        churn_risk = 'churn' if pred == 1 else 'no churn'
        yes_percent = prob[1] * 100
        no_percent = prob[0] * 100

        flash('Percentage of this customer leaving is: %.0f%%' % yes_percent)

        return render_template(
            'score.html',
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
