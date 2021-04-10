import numpy as np
import pandas as pd


from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, recall_score, classification_report, accuracy_score, f1_score
    
from sklearn.model_selection import KFold



def model_evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    
    ass = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return cm, aps, ps, rs, ass, f1s, class_report

# Cross validation function
def cross_validation_fun(k, training_data, test_data, model):
    kf = KFold(n_splits = k, random_state = None)
    X_test_data = test_data.iloc[:,:-1]
    y_test_data = test_data.iloc[:, -1]
    acc_score =[]
    aps_score =[]
    rs_score =[]
    f1s_score =[]
    
    # k-fold cross validation was applied to the original data
    for train_index, test_index in kf.split(X_test_data):
        X_test, y_test = X_test_data.iloc[test_index,:], y_test_data[test_index]
        
        # one fold of the original dataset was considered as the test set
        test_data_part = X_test.join(y_test)
        training_data_temp = training_data
        
        # all the instances of this fold was removed from the preprocessed version of data. the remaining instances as considered 
        # training data for that iteration 
        training_data_part = training_data_temp[~training_data_temp.isin(test_data_part)].dropna()
        X_train = training_data_part.iloc[:,:-1]
        y_train = training_data_part.iloc[:,-1]
        
        # model fitted using training dataset and prediction made on test set
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        
        cm, aps, ps, rs, ass, f1s, class_report = model_evaluation(y_test, pred_values)
        
        # calculation of four performance metrics made for each iteration
        acc= accuracy_score(pred_values, y_test)
        acc_score.append(acc)
        aps_score.append(aps)
        rs_score.append(rs)
        f1s_score.append(f1s)
    
    # Average of all metrics being calculated
    
    avg_aps_score = sum(aps_score)/k   
    avg_acc_score = sum(acc_score)/k
    avg_rec_score = sum(rs_score)/k
    avg_f1_score = sum(f1s_score)/k
    print("Avg accuracy test : {}".format(avg_acc_score) )
    print("Avg precision score : {}".format(avg_aps_score) )
    print("Avg recall score : {}".format(avg_rec_score) )
    print("Avg f1 score : {}".format(avg_f1_score) )
    return avg_acc_score, avg_aps_score
    
 

    