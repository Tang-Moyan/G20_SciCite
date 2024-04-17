import subprocess
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Local_Predictor import LocalPredictor

predictor = LocalPredictor(f"{os.path.dirname(__file__)}/../../data/scicite.tar.gz")

def retrieve_allennlp_predictions(id_string_list):
    '''
    This function takes in data of the form (id, string) and tries to predict if it is a background, method or result using the allennlp model.
    
    It will then return the predictions in the form of (id, prediction) where prediction is either "background", "method" or "result".
    '''
    return [(id, predictor.predict(string)) for id, string in id_string_list]

def retrieve_allennlp_prediction(string):
    return predictor.predict(string)