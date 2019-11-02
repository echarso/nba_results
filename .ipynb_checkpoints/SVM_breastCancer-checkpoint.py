#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
from sklearn import datasets
import numpy as np
import pandas as pd


#Load dataset
cancer = datasets.load_breast_cancer()
# print the names of the 13 features

print(type(cancer.data))
print(type(cancer.feature_names))
print(type(cancer.target_names))

df = pd.read_csv('games_records.csv')
print(df);
df1 = df.iloc[1:,-2]
df1.to_csv('again.csv')


print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print data(feature)shape
print ( cancer.data.shape)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])
print(cancer.target)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print(X_train)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(y_train)


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
