import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

'''
Imagine a telecommunications provider has segmented its customer base by service 
usage patterns, categorizing the customers into four groups. If demographic data 
can be used to predict group membership, the company can customize offers for 
individual prospective customers. It is a classification problem. That is, given 
the dataset, with predefined labels, we need to build a model to be used to predict 
class of a new or unknown case.

The example focuses on using demographic data, such as region, age, and marital, 
to predict usage patterns.

The target field, called custcat, has four possible values that correspond to the 
four customer groups, as follows: 1- Basic Service 2- E-Service 3- Plus Service 4- 
Total Service

Our objective is to build a classifier, to predict the class of unknown cases. 
We will use a specific type of classification called K nearest neighbour.
'''

## Reading the data
df = pd.read_csv('teleCust1000t.csv')

## Adjust display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# print(df.head())

## How many of each class is in our data set
# print(df['custcat'].value_counts())
## 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

## Visualizing data
# df.hist(column='income', bins=50)
# plt.show()

## Defining feature sets
print(df.columns)

## To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
print(X[0:5]) # To show first 5 rows

y = df['custcat'].values
print(y[0:5])
print(type(X))
'''
Data Standardization gives the data zero mean and unit variance, 
it is good practice, especially for algorithms such as KNN which 
is based on the distance of data points:
'''
## Create a StandardScalar object
# scaler = preprocessing.StandardScaler()
# Fit the scalar to the data and transorm the data
# x = scaler.fit_transform(X.astype(float))
## Or combine
x = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(x[0:5])

## Train Test Split
'''
Out of Sample Accuracy is the percentage of correct predictions 
that the model makes on data that the model has NOT been trained on. 
Doing a train and test on the same dataset will most likely have low 
out-of-sample accuracy, due to the likelihood of our model overfitting.

It is important that our models have a high, out-of-sample accuracy, 
because the purpose of any model, of course, is to make correct predictions 
on unknown data. So how can we improve out-of-sample accuracy? One way is to 
use an evaluation approach called Train/Test Split. Train/Test Split involves 
splitting the dataset into training and testing sets respectively, which are 
mutually exclusive. After which, you train with the training set and test with 
the testing set.

This will provide a more accurate evaluation on out-of-sample accuracy because 
the testing dataset is not part of the dataset that has been used to train the 
model. It is more realistic for the real world problems.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test_set:', X_test.shape, y_test.shape)

## Classification (KNN)
## Training model with k = 4
k = 4
neigh4 = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh4)

## Predicting
yhat4 = neigh4.predict(X_test)
print(yhat4[0:5])

## Accuracy evaluation
'''
In multilabel classification, accuracy classification score is a function that 
computes subset accuracy. This function is equal to the jaccard_score function. 
Essentially, it calculates how closely the actual labels and predicted labels 
are matched in the test set.
'''
print("Train set Accuracy (k = 4): ", metrics.accuracy_score(y_train, neigh4.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat4))

## Training model with k = 6
k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh6)
yhat6 = neigh6.predict(X_test)
print(yhat6[0:5])

print("Train set Accuracy (k = 6): ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

'''
What about other K?
K in KNN, is the number of nearest neighbors to examine. It is supposed to be 
specified by the user. So, how can we choose right value for K? The general 
solution is to reserve a part of your data for testing the accuracy of the model. 
Then choose k =1, use the training part for modeling, and calculate the accuracy 
of prediction using all samples in your test set. Repeat this process, increasing 
the k, and see which k is the best for your model.

'''
## Calculating the accuracy of KNN for different values of k
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

## Plot the model accuracy for a different number of neighbours
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha = 0.1)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha = 0.1, color = "green")
plt.legend(('Accuracy', '+/- 1xstd', '+/-3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)
