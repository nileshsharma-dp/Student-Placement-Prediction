from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

## Terminal Codes
# python -m venv my_vertual_env
# my_vertual_env\Scripts\activate
# pip install flask gunicorn numpy pandas scikit-learn
# pip show flask numpy pickle scikit-learn gunicorn pandas
# pip install flask numpy scikit-learn gunicorn pandas
# pip freeze > requirements.txt
# python app.py



# Load the model
model_path = 'model.pkl'
with open(model_path,'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index2.html')

# Define the predict route

# @app.route('/predict',methods=['POST'])
# def predict():
#     # Get the input values
#     # int_features = [int(x) for x in request.form.values()]
#     # final_features = [np.array(int_features)]

#     # Make prediction
#     pridiction = model.predict(final_features)
#     output = 'placement' if pridiction[0] == 1 else 'no placement'

#     return render_template('index2.html',prediction_text='Student will get {}'.format(output))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert all input values to float
        float_features = [float(x) for x in request.form.values()]
        prediction = model.predict([float_features])

        # The result is displayed after prediction
        return render_template('index2.html', prediction_text=f'Placement Prediction: {prediction[0]}')
    except Exception as e:
        return f"An error occurred: {e}"



    # cgpa = request.form.get('cgpa')
    # iq = request.form.get('iq')
    # profile_score = request.form.get('profile_score')
    # input_query = np.array([[cgpa,iq,profile_score]])
    # result = model.predict(input_query)[0]
    # return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug=True) 