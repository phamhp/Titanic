
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from random import randint,uniform
from pandas.tests.test_msgpack.test_subtype import MyList
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC   
from sklearn.neighbors import KNeighborsClassifier


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
 
    plt.legend(loc="best")
    return plt


 

train_url = "/tmp/titanic_data.csv"
#create data frame object
df = pd.read_csv(train_url)

df = df.drop(['Ticket','Cabin'],axis = 1)

labelEncoder = preprocessing.LabelEncoder()
df['Embarked'] = labelEncoder.fit_transform(df['Embarked'])
df['Sex'] = labelEncoder.fit_transform(df['Sex'])
# convert values of columns to numbers : 0,1,2,3,4...etc
labelEncoder = preprocessing.Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
df['Age'] = labelEncoder.fit_transform(df[['Age']]).ravel()
labelEncoder = preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)
df['Embarked'] = labelEncoder.fit_transform(df[['Embarked']]).ravel()
df['Fare'] = labelEncoder.fit_transform(df[['Fare']]).ravel()  
    #train, test = train_test_split(data, test_size = 0.2,random_state = randint(0,100)) 
y = df['Survived']
x = df[['Pclass','Sex','Age','Fare']]
 
title = "Learning Curves (Decision Tree)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=40)
estimator = DecisionTreeClassifier(max_depth=10)
plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
   
title = "Learning Curves (Neural Network)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=40)
estimator = MLPClassifier(max_iter=2000)
plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
  
title = "Learning Curves (SVM)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=40)
estimator = SVC()
plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
  
title = "Learning Curves (K-NN)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=40)
estimator = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
  
title = "Learning Curves (GradientBoosting Decision Tree)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=40)
estimator = GradientBoostingClassifier(random_state=0)
plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
   
plt.show()


