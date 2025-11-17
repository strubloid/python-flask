from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle

# Create the Flask application instance
app = Flask(__name__)


## loading the model example_weights_knn.pkl
model = pickle.load(open('example_weights_knn.pkl', 'rb'))

# Define a route for the root URL
@app.route('/')
def use_template():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_one=request.form['1']
    input_two=request.form['2']
    input_three=request.form['3']
    input_four=request.form['4']
    input_five=request.form['5']
    input_six=request.form['6']
    input_seven=request.form['7']
    input_eight=request.form['8']

    setup_df = pd.DataFrame([pd.Series([input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight])])
    diabetes_prediction = model.predict(setup_df)
    output='{0:.{1}f}'.format(diabetes_prediction[0][1],2)
    output = str(float(output)*100) + " %"

    if output>str(0.5):
        return render_template('result.html', pred=f'You hae the following change of having diabetes: {output}')
    else:
        return render_template('result.html', pred=f'You have a low chance of having diabetes: {output}')


# Another way to do this
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)