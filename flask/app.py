from copyreg import pickle
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

loaded_model = joblib.load(open("model.pkl", "rb"))



@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    Open = request.form.get('Open')
    High = request.form.get('High')
    Low = request.form.get('Low')

    prediction = loaded_model.predict([[Open, High, Low]])
    if prediction[0] == 1:
        val = 'Prediction: Stock will go up'
    else:
        val = 'Prediction: Stock will go down'

    return render_template('result.html' , value = val)

    


if __name__ == '__main__':
 app.run(debug=True)