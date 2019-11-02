#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import svm

#http://www.handsonmachinelearning.com/blog/McTKK/python-one-hot-encoding-with-scikit-learn

#Load dataset
#https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# print the names of the 13 features




df = pd.read_csv('my_game_records.csv')

trgt = df['target_flag']
df.drop(['date','h_team','v_team','target_flag','d_name','datetime'], 1, inplace=True)
df = df.fillna(0)

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
print('splitting data set.....');
X_train, X_test, y_train, y_test = train_test_split(df.values,
                                                    trgt,
                                                    test_size=0.3,
                                                    random_state=109) # 70% training and 30% test



print('splitting done.....');
print (df.columns.values.tolist())
#Create a svm Classifier
print('fit.....');
from sklearn.svm import LinearSVC
print (X_train.shape)

clf =LinearSVC().fit(X_train, y_train)
print('fit  done.....');

#Predict the response for test dataset
print('predict .....');
y_pred = clf.predict(X_test)
print('predict  done.....');

from sklearn import metrics
from sklearn.metrics import roc_auc_score

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print ('roc_auc_score:', roc_auc_score(y_test, y_pred))
'''
print('Real prediction .....');
print('1.Import data from games_predict file.');


df = pd.read_csv('games_predict.csv')

data = df.values
print("1=>Imported data\n" ,data);
print("2.One hot encoder done ");
print(df.values);
print("3.Predict")
y_real_predictions = clf.predict(df.values);
print(y_real_predictions);
'''
