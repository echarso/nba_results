#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm


#http://www.handsonmachinelearning.com/blog/McTKK/python-one-hot-encoding-with-scikit-learn

#Load dataset
#https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# print the names of the 13 features




df = pd.read_csv('my_game_records.csv')

df['sal_diff']=df['sal_diff'].abs()
df['in_win_diff']=df['in_win_diff'].abs()
df['def_vs_offs']=df['defence_scored_average']- df['points_scored_average']
df['def_vs_offs']=df['def_vs_offs'].abs()
def_vs_offs=0

TOTAL_THREASHOLD = 200
for i in range(0,df.shape[0]):
    def_vs_offs = df.at[i,'defence_scored_average']- df.at[i,'points_scored_average']
    def_vs_offs= abs(def_vs_offs)

    if df.at[i,'sal_diff'] < 10000 or df.at[i,'in_win_diff']<3 or def_vs_offs >20:
        print( 'no row in ', df.at[i,'in_win_diff'],def_vs_offs)
        df = df.drop(i)
    else:
        if df.at[i,'total_points'] > TOTAL_THREASHOLD:
            df.at[i,'target_flag']=1
        else:
            df.at[i,'target_flag']=0

df_positive = df[df['target_flag']==1]
df_negative = df[df['target_flag']==0]
df['sal_diff']=df['sal_diff'].abs()

df['in_win_diff']=df['in_win_diff'].abs()
df['out_win_diff']=df['out_win_diff'].abs()
df['difference']=df['difference'].abs()
print('========================================' , df.shape[0])
df = df.iloc[50:,:]
trgt = df['target_flag']
print (len(df_positive) /float(len(df_negative)))
print('========================================' , df.shape[0])
df = df.fillna(0)
df = df[['defence_scored_average','points_scored_average','in_win_diff','out_win_diff' ,'def_vs_offs'  ]]
print (df.head(10))

from sklearn.model_selection import train_test_split
print('splitting data set.....');

X_train, X_test, y_train, y_test = train_test_split(df.values,
                                                    trgt,
                                                    test_size=0.3,
                                                    random_state=109) # 70% training and 30% test



print('splitting done.....');


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri('http://127.0.0.1:5000');

#Create a svm Classifier
C_params=[0.1,0.5,0.8,1,3,4,10,20]
max =0
c_max = 0
d_max = 0
ac_max=0

clf =LinearSVC().fit(X_train, y_train)
y_pred = clf.predict(X_test)

tmp =metrics.roc_auc_score(y_test, y_pred);
accuracy = metrics.accuracy_score(y_test, y_pred)

mlflow.log_metric("roc ", tmp )
mlflow.log_metric('accuracy ' , accuracy)

mlflow.sklearn.log_model(clf, "model LinearSVC 1")

print ('------------------------------------');
