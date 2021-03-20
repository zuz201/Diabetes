#import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score
    



def model_evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    
    ass = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return cm, aps, ps, rs, ass, f1s, class_report