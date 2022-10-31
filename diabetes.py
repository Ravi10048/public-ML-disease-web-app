import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm #support vector machine (important algo in supervised learning,in these i train my model with fetures like blood glucose level ,insulin level and label- diabetic or not,once we feed this dataset to this model and its tries to plot the data  and once its plot the data its tries to find a hyper plane plane which separtes two group and whenever when we input a new data it will tries to put that data in either of two groups and by that it can predict whether the person is diabetic or not)
from sklearn.metrics import accuracy_score # used for classifier models
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
# loading the diabetes dataset to a pandas DataFrame
#UCI ML repo
diabetes_dataset = pd.read_csv('diabetes.csv') 
# printing the first 5 rows of the dataset
print(diabetes_dataset.info())
print(diabetes_dataset.head())

# number of rows and Columns in this dataset
print(diabetes_dataset.shape)

# getting the statistical measures of the data
print(diabetes_dataset.describe())

# outcome is label in this model
#0 --> Non-Diabetic
#1 --> Diabetic
print(diabetes_dataset['Outcome'].value_counts())

'''0    500
   1    268
Name: Outcome, dtype: int64'''

print(diabetes_dataset.groupby('Outcome').mean())


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1) # (in diabetes dataset i dropping the outcome column,1-column)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
# Data standardization (if there is a diiference in the range of all these value it will be difficult for our model to predict the data we should standarize the data that helps our ml model to make better prediction )
scaler=StandardScaler()
scaler.fit(X) # we are now fitting all the unstandardizes data 
standardized_data=scaler.transform(X)
print(standardized_data)

X=standardized_data
Y=diabetes_dataset['Outcome'] 

#Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2) # we stratify with respect to y so that our y data evenly distributed into train and test  
print(X.shape, X_train.shape, X_test.shape)

#Creating and Training the Model

classifier = svm.SVC(kernel='linear') # creating model (svm as linear model)
classifier.fit(X_train, Y_train) # training the model 

# accuracy score on the training data
Y_train_prediction = classifier.predict(X_train) 
training_data_accuracy = accuracy_score(Y_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)# 0.7833876221498371

# accuracy score on the test data
Y_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)# 0.7727272727272727
# plt.plot(Y_test,Y_test_prediction)
# plt.show()
 
#Making a Predictive System

import pickle # use to make file
import numpy as np
# The pickle module is used for implementing binary protocols for serializing and de-serializing a Python object structure. 

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb')) #This function is called to de-serialize a data stream.

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic') 
 