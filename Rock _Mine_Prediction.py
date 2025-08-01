#importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading data
sonar_data=pd.read_csv("/content/sonar data.csv",header=None)

sonar_data.groupby(60).mean()

#seperating data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]

#training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

#model training ->logistic regression
model=LogisticRegression()

#training logistic regression model with training data
model.fit(X_train,Y_train)

#model evaluation

#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print(training_data_accuracy)

#accuracy on test data
X_test_prediction=model.predict(X_test)
training_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print(training_data_accuracy)

#Making a predictive system

input_data=(0.01,0.0171,0.0623,0.0205,0.0205,0.0368,0.1098,0.1276,0.0598,0.1264,0.0881,0.1992,0.0184,0.2261,0.1729,0.2131,0.0693,0.2281,0.406,0.3973,0.2741,0.369,0.5556,0.4846,0.314,0.5334,0.5256,0.252,0.209,0.3559,0.626,0.734,0.612,0.3497,0.3953,0.3012,0.5408,0.8814,0.9857,0.9167,0.6121,0.5006,0.321,0.3202,0.4295,0.3654,0.2655,0.1576,0.0681,0.0294,0.0241,0.0121,0.0036,0.015,0.0085,0.0073,0.005,0.0044,0.004,0.0117)

#changing to numpy array
input_data_as_numpy=np.asarray(input_data)

#reshape the np array as we are predicting for one instace
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

if(prediction=='R'):
  print("its ROCK")
else:
  print("its MINE")
