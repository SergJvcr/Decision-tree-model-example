# Standard operational package imports
import pandas as pd
import numpy as np
# Important imports for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.metrics as metrics
# Visualization package imports
import matplotlib.pyplot as plt
import seaborn as sns

df_original = pd.read_csv("google_data_analitics\\Invistico_Airline.csv")

print(df_original.head(10))

# Data exploration, data cleaning, and model preparation
# Exploring the data
print(df_original.info())
print(df_original.dtypes)
print(df_original.describe(include='all'))

# Output unique values
print('Unique labels for Class feature:', df_original['Class'].unique())

# Check the counts of the predicted labels
print(df_original['satisfaction'].count())
print(df_original['satisfaction'].value_counts())
print(f'Here is {round((df_original["satisfaction"].value_counts()[0]  / df_original["satisfaction"].count()) * 100, 2)}% of satisfied people')

# Check for missing values
print(df_original.isna().sum())

# Check the number of rows and columns in the dataset
print(f'Here are {df_original.shape[0]} rows and {df_original.shape[1]} columns.')

# Drop the rows with missing values
df_subset = df_original.dropna(axis=0).reset_index(drop=True)

# Check for missing values after cleaning
print(df_subset.isna().sum())

# Check the number of rows and columns in the dataset again
print(f'Here are {df_subset.shape[0]} rows and {df_subset.shape[1]} columns after the empty data was deleted.')

# Encode the data
# Four columns (satisfaction, Customer Type, Type of Travel, Class) 
# are the pandas dtype object. Decision trees need numeric columns.
df_subset['Class'] = df_subset['Class'].replace({'Business':3, 'Eco Plus':2, 'Eco':1})

# Represent the data in the target variable numerically
df_subset['satisfaction'] = df_subset['satisfaction'].replace({'satisfied':1, 'dissatisfied':0})

# Convert categorical columns into numeric
df_subset = pd.get_dummies(df_subset, drop_first=True) #(Customer Type, Type of Travel)

# Check column data types
print(df_subset.dtypes)
print(df_subset.head(10)) # or df_subset.tail(10)

# Create the training and testing data
y = df_subset['satisfaction']
X = df_subset.copy()
X = df_subset.drop('satisfaction', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# Model building
# Fit a decision tree classifier model to the data
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
df_pred = decision_tree.predict(X_test)

# Results and evaluation
print("The decision tree model's accuracy, precision, recall, and F1 scores:")
print(f'Accuracy: {round(metrics.accuracy_score(y_test, df_pred), 3)}')
print(f'Precision: {round(metrics.precision_score(y_test, df_pred), 3)}')
print(f'Recall: {round(metrics.recall_score(y_test, df_pred), 3)}')
print(f'F1 score: {round(metrics.f1_score(y_test, df_pred), 3)}')

# Produce a confusion matrix (to know the types of errors made by an algorithm)
conf_matrix = metrics.confusion_matrix(y_test, df_pred, labels=decision_tree.classes_)
display_conf_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=decision_tree.classes_)
display_conf_matrix.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.show()

# Plot the decision tree (to examine the decision tree)
names_for_features = list(X.columns)
# names_for_classes = {0:'dissatisfied', 1:'satisfied'}
names_for_classes = ['dissatisfied', 'satisfied']
plt.figure(figsize=(30, 12))
plot_tree(decision_tree=decision_tree, max_depth=3, fontsize=7, feature_names=names_for_features,
         class_names=names_for_classes, filled=True)
plt.show()

# Using the feature_importances_ attribute to fetch the relative importances of each feature
importances = decision_tree.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax, color='orange')
ax.set_title('the relative importances of each feature')
plt.show()

# Hyperparameter tuning
tree_para = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
             'min_samples_leaf': [2,3,4,5,6,7,8,9, 10, 15, 20, 50]}

scoring = ['accuracy', 'precision', 'recall', 'f1']

# Check combinations of values
# Check every combination of values to examine which pair has the best evaluation metrics
tuned_decision_tree = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(tuned_decision_tree,
                  tree_para,
                  scoring=scoring,
                  cv=5,
                  refit='f1')
clf.fit(X_train, y_train)

# Compute the best combination of values for the hyperparameters
print(f'The best combination of values for the hyperparameters is {clf.best_estimator_}')
print(f'The best Average Validation Score is {round(clf.best_score_, 3)}')

# Determine the "best" decision tree model's accuracy, precision, recall, and F1 score

results = pd.DataFrame(columns=['Model', 'F1', 'Recall', 'Precision', 'Accuracy'])

def make_results(model_name, model_object):
    """
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.  
    """

    # Get all the results from the CV and put them in a df.
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score).
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row.
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create a table of results.
    #table = pd.DataFrame()
    #table_ = table_.append({'Model': model_name,
    #                      'F1': f1,
    #                      'Recall': recall,
    #                      'Precision': precision,
    #                      'Accuracy': accuracy},
    #                     ignore_index=True)
    table_ = {'Model': model_name,
              'F1': f1,
              'Recall': recall,
              'Precision': precision,
              'Accuracy': accuracy}
    table = pd.DataFrame.from_dict(data=table_, orient='index', columns=[''])
    table.transpose()

    return table

result_table = make_results("Tuned Decision Tree", clf)

print(result_table)

# Plot the "best" decision tree
names_for_features = list(X.columns)
names_for_classes = ['dissatisfied', 'satisfied']
plt.figure(figsize=(30, 12))
plot_tree(decision_tree=clf.best_estimator_, max_depth=3, fontsize=7, feature_names=names_for_features,
         class_names=names_for_classes, filled=True)
plt.show()

# Using the feature_importances_ attribute to fetch the relative importances of each feature
importances = clf.best_estimator_.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax, color='lightblue')
ax.set_title('the relative importances of each feature')
plt.show()
