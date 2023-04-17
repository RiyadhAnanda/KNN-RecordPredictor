# KNN- Predicting whether a college football team will have a winning/losing record

# Imports
import pandas as pd
import numpy as np

# Import various functions that will be used later from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Import data - contained in mlpracdata1.csv
data = pd.read_csv('mlpracdata1.csv')
# Print the total number of data entries in dataset
print('Number of Data entries:',len(data) )
# Print the first 5 data points using head function 
print( data.head() )


# Columns 1-15 will correspond to X data points, column 0 has school name so we wont need this one
X = data.iloc [:, 1:16]
# Column 16 contains 'Record' meaning if 1 the team had a winning record, if 0 the team did not have a winning record
y = data.iloc[:, 16]
# Split the data into training and testing set, the test set will contain 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size =.2)

# sklearn: scaler function. See reference [3] from report
# Scale the data 
scale_X = StandardScaler()
# Fit with training set
X_train = scale_X.fit_transform(X_train)
# Transform testing set
X_test = scale_X.transform(X_test)

# Take the sqrt root of the length of the test set, this is standard practice for identifying k value
import math
# Print the sqrt, this will be 5.196 at is the sqrt of 20% of the 131 (26.2-rounds up to 27 entries) total entries
print ('Suggested k-score(round to nearest odd integer):',math.sqrt(len(y_test)))

# sklearn: KNeighborsClassifier function. See reference [1] in report
# Define the KNN model using KNeighborsClassifier, n_neighbors corresponds to the k value, p corresponds to number of outcomes
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
#sklearn: fitting and testing of data. See reference [2] in report
# Fit the KNN model
classifier.fit(X_train, y_train)
# Predict results of test set
y_pred = classifier.predict(X_test)
y_pred

#sklearn: Confusion_matrix function. See reference [4] in report
# Use the confusion_matrix function to evaluate model and set it equal to variable 'conmat'
conmat = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:")
# Print confusion matrix
print (conmat)

#sklearn: f1_score function. See reference [5] in report
# Use f1_score function to find the f1 score of the model
print ('F1 Score:', f1_score(y_test, y_pred))

#sklearn: accuracy_score function. See reference [6] in report
# Use accuracy_score function to find the accuracy of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
