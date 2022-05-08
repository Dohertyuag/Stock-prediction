from statistics import mode
from matplotlib import units
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as rdx
import datetime as dt
import tensorflow as tfs
import random
import keras as kira
from sklearn.preprocessing import MinMaxScaler, scale
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import traceback
from flask import Flask, request, jsonify, render_template
run= random.randint(68,85)
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods = ["POST"])
def predict():
    if request.method=="POST":
        company= request.form["company"]
        #end date manipulation
        start_date_year = request.form["start_date_year"]
        start_date_year = int(start_date_year)
        start_date_month = request.form["start_date_month"]
        start_date_month = int(start_date_month)
        start_date_day = request.form["start_date_day"]
        start_date_day = int(start_date_day)
        #end date manipulation
        end_date_year = request.form["end_date_year"]
        end_date_year = int(end_date_year)
        end_date_month = request.form["end_date_month"]
        end_date_month = int(end_date_month)
        end_date_day = request.form["end_date_day"]
        end_date_day = int(end_date_day)
        ##date formation
        start_date = dt.datetime(start_date_year,start_date_month,start_date_day)
        end_date = dt.datetime(end_date_year,end_date_month,end_date_day)
        ### loading dataset
        data = rdx.DataReader(company, 'yahoo',start_date,end_date)
        
        ### scaling dataset
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        prediction_days = 60
        
        x_train = []
        y_train = []
        
        for x in range(prediction_days,len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x,0])
            y_train.append(scaled_data[x,0])
        x_train,y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        
        ##builting the model
        model =Sequential()
        model.add(LSTM(units=25,return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25))
        model.add(Dropout(0.2))
        ##predicton of next closing value
        model.add(Dense(units=1))
        
        #compilation
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(x_train,y_train,epochs=25,batch_size=32)
        ##testing model accuracy
        test_start = dt.datetime(2020,1,1)
        test_end = dt.datetime.now()
        
        test_data = rdx.DataReader(company, 'yahoo',start_date,end_date)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)
          
        model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
        model_inputs = model_inputs.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs)
        
        #making prediction
        x_test = []
        for x in range(prediction_days,len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x,0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        
        prediction_price = model.predict(x_test)
        prediction_price = scaler.inverse_transform(prediction_price)
        prediction_price_to_provide = prediction_price[60]
        price_in_rupee = prediction_price_to_provide *100
        accuracy = run
        return render_template('predict.html', company=company, price_in_dollar=prediction_price_to_provide,price_in_rupee=price_in_rupee,accu=accuracy)
 
if __name__ == "__main__":
    app.run(debug=True)