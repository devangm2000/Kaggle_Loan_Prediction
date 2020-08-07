# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:35:50 2020

@author: Devang Mehrotra
"""


from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle


app =Flask(__name__)


pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template("index.html")
    

@app.route('/predict',methods=['POST'])
def predict(): 

    '''
    For rendering results on HTML GUI 2 6 7 8 10
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array([features[1],features[5],features[6],features[7],features[9]])]
    prediction=classifier.predict(final_features)
    return render_template('index.html', prediction_text='The output is {}'.format(prediction))


if __name__ == "__main__":
    app.run(port=5000,debug=True)
