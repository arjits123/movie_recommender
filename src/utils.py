import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np # type: ignore
import pandas as pd # type: ignore
import ast
import dill # type: ignore
from nltk.stem.porter import PorterStemmer

def save_obj(file_path, data_frame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(data_frame, f)
    except Exception as e:
        pass

def convert(obj):
    new_obj = ast.literal_eval(obj)
    mylist = []
    for i in new_obj:
        value = i['name']
        mylist.append(value)
    return mylist

def convert_cast(obj):
    new_obj = ast.literal_eval(obj)
    mylist = []
    counter = 0
    for i in new_obj:
        if counter != 5:   
            value = i['name']
            mylist.append(value)
            counter += 1
        else:
            break
    return mylist

def convert_crew(obj):
    mylist = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            mylist.append(i['name'])
            break
    return mylist

ps = PorterStemmer()
def stemming(text):
    
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
        