from logging import debug
from flask import Flask,request,render_template
import pickle
import numpy as np
import os
import config
app = Flask(__name__)

@app.route('/')  # base API
def welcome():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def prediction():
    data = request.form

    Age = float(data['Age'])
    Sex = float(data['Sex'])
    ChestPainType = float(data['ChestPainType'])
    Cholesterol = float(data['Cholesterol'])
    FastingBS = float(data['FastingBS'])
    MaxHR = float(data['MaxHR'])
    ExerciseAngina = float(data['ExerciseAngina'])
    Oldpeak = float(data['Oldpeak'])
    ST_Slope = float(data['ST_Slope'])

    model_folder_path = config.MODEL_FOLDER_PATH
    xgb_model = pickle.load(open(f'{model_folder_path}/pickel_file.pkl','rb'))
        
    data_ = np.array([[Age, Sex, ChestPainType, Cholesterol, FastingBS,MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
    result = xgb_model.predict(data_)
    if result[0] == 1:
        result = "High Chances of Heart Fail. Do checkup and exercise regularly"
    else:
        result ="Less Chances of Heart Fail. Do exercise regularly"

    return render_template('index.html',prediction = result)

       
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)