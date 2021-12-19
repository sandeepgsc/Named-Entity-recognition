import numpy as np
from flask import Flask, request, jsonify, render_template
import simpletransformers
import pandas as pd
import requests
from simpletransformers.ner import NERModel
import json
import io
import torch

import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', "GET"])
def predict():
    int_features = [x for x in request.form.values()]
    print(int_features)

    text = int_features[0]
    model_bert = open('NER_model.pkl', 'rb')
    model_bert_2 = pickle.load(model_bert)
    text = text.upper()

    predictions, raw_outputs = model_bert_2.predict([text])
    perfect_words = {}
    meaningless_words = []
    if int_features[1] == 'kamal':
        for t in predictions:
            for i in range(len(t)):
                for l, k in t[i].items():
                    if k == 'O':
                        meaningless_words.append(l)
                    else:
                        perfect_words[l] = k

        result = json.dumps(perfect_words)
        return render_template('index.html', prediction_text=result)


if __name__=='__main__':
    app.run(debug=True)
