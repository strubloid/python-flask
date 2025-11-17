from flask import Flask, jsonify, request, url_for, redirect, render_template
import pandas as pd
import pickle
from flask_cors import CORS

# Create the Flask application instance
app = Flask(__name__)

# adding the CORS middleware
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://ds-frontend-8b124c0b64a5.herokuapp.com",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

## loading the model example_weights_knn.pkl
model = pickle.load(open('example_weights_knn.pkl', 'rb'))

# Validation function
def validate_form_data(form_data):
    """
    Validate that all required fields are present and not empty.
    Returns (is_valid, error_message)
    """
    required_fields = [
        'pregnancies', 'glucose', 'blood_presure', 'skin_thickness',
        'insulin_level', 'bmi', 'diabetes_pedigree', 'age'
    ]
    
    # Check if all fields are present and not empty
    for field in required_fields:
        if field not in form_data or not form_data[field].strip():
            return False, f"Field '{field}' is required and cannot be empty"
    
    return True, None


# Define a route for the root URL
@app.route('/')
def use_template():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():

    # Validate form data
    is_valid, error_message = validate_form_data(request.form)
    
    ## Basic validation
    if not is_valid:
        return render_template('result.html', pred=f'Error: {error_message}'), 400
   
    input_one = request.form['pregnancies']
    input_two = request.form['glucose']
    input_three = request.form['blood_presure']
    input_four = request.form['skin_thickness']
    input_five = request.form['insulin_level']
    input_six = request.form['bmi']
    input_seven = request.form['diabetes_pedigree']
    input_eight = request.form['age']
    
    # Create DataFrame with proper column names (8 features for diabetes dataset)
    setup_df = pd.DataFrame([[input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight]])
    
    # Get probability predictions instead of just class prediction
    diabetes_prediction = model.predict_proba(setup_df)
    
    # Get probability of positive class (diabetes)
    probability = diabetes_prediction[0][1]
    output = '{0:.{1}f}'.format(probability * 100, 2)

    return jsonify({
        'prediction': 'positive' if probability > 0.5 else 'negative',
        'probability': float(output),
        'probability_formatted': output + '%',
        'message': f'You have {"a high" if probability > 0.5 else "a low"} chance of having diabetes: {output}%'
    })

    # if probability > 0.5:
    #     return render_template('result.html', pred=f'You have the following chance of having diabetes: {output}')
    # else:
    #     return render_template('result.html', pred=f'You have a low chance of having diabetes: {output}')


# Another way to do this
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)