import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn import metrics
import pydotplus
import sklearn.tree as tree
import matplotlib.pyplot as plt

### In order to plot the tree pydotplus and graphviz libraries must be installed

'''
Imagine that you are a medical researcher compiling data for a study. 
You have collected data about a set of patients, all of whom suffered 
from the same illness. During their course of treatment, each patient 
responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and Drug y.

Part of your job is to build a model to find out which drug might be 
appropriate for a future patient with the same illness. The features 
of this dataset are Age, Sex, Blood Pressure, and the Cholesterol of 
the patients, and the target is the drug that each patient responded to.

It is a sample of multiclass classifier, and you can use the training 
part of the dataset to build a decision tree, and then use it to predict 
the class of an unknown patient, or to prescribe a drug to a new patient.
'''
my_data = pd.read_csv('drug200.csv', delimiter=",")
print(my_data.head())
print(my_data.shape)

## X as the Feature Matrix (data of my_data)
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol',  'Na_to_K']].values
print(X[0:5])

### Pre-processing

'''
Some features in this dataset are categorical, such as Sex or BP. 
Sklearn Decision Trees does not handle categorical variables. 
We can still convert these features to numerical values using 
the LabelEncoder() method to convert the categorical variable 
into dummy/indicator variables.
'''

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

## y as the response vector (target)
y = my_data['Drug']
print(y[0:5])


### Setting up the Decision Tree
'''
We will be using train/test split on our decision tree. 

train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split, the test_size represents the 
ratio of the testing dataset, and the random_state ensures that we obtain the same splits.
'''

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

## Printing the shape to ensure that the dimensions match
print(f'Shape of the X training set {X_trainset.shape} & Size of the y training set {y_trainset.shape}')
print(f'Shape of the X test set {X_testset.shape} & Size of the y test set {y_testset.shape}')

### Modeling

'''
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
'''
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree.get_params()) # Print the default parameters

## Fitting the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset, y_trainset)

### Prediction

predTree = drugTree.predict(X_testset) ## prediction on the testing dataset

print(predTree[0:5])
print(y_testset[0:5])

### Evaluation
'''
import metrics from sklearn and check the accuracy of our model.

Accuracy classification score computes subset accuracy: the set of labels predicted for a 
sample must exactly match the corresponding set of labels in y_true.

In multilabel classification, the function returns the subset accuracy. If the entire set of 
predicted labels for a sample strictly matches with the true set of labels, then the subset 
accuracy is 1.0; otherwise it is 0.0.
'''

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

### Visualizing the decision tree
## Export the decision tree to a dot file
dot_data = export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

## Generate the visualization
graph = pydotplus.graph_from_dot_file('tree.dot')
# graph.write_png('decosion_tree.png')

## Show the plot
plt.figure(figsize=(20, 20))
plot_tree(drugTree, filled= True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], class_names=drugTree.classes_)
# plt.axes('off')

## Save the visualization to tree.png
plt.tight_layout()
plt.savefig('decision_tree.png')

## Show the plot
plt.show()
