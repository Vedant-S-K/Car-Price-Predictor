from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import os


app = Flask(__name__, template_folder="template")
cors = CORS(app)
car = pd.read_csv('Cleaned_Car_data.csv')
model = pickle.load(open("./model/LinearReg.pkl","rb"))
print("Model Loaded")

@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique(), reverse=True)

    return render_template("predict.html", companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

    
@app.route("/predict", methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')
        print(car_model,company,year,driven,fuel_type)
        
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    data = np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))

if __name__=='__main__':
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0" , port=port)
