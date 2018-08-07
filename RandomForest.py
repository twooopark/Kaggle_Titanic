import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
'''
Survived: 0 = No, 1 = Yes
pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
sibsp: # of siblings / spouses aboard the Titanic
parch: # of parents / children aboard the Titanic
ticket: Ticket number
cabin: Cabin number
embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
'''
# Printing first 5 rows of the train dataset.
train.head()
test.head()

train.shape
test.shape


# We can see that Age value is missing for many rows.
# Out of 891 rows, the Age value is present only in 714 rows.
# Similarly, Cabin values are also missing in many rows. Only 204 out of 891 rows have Cabin values
train.isnull().sum()
test.isnull().sum()


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


train.head()
test.head()

# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# male: 0, female: 1
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train.head()
test.head()


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

'''
Binning/Converting Numerical Age to Categorical Variable  

feature vector map:  
child: 0  
young: 1  
adult: 2  
mid-age: 3  
senior: 4
'''
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 18, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 35), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 43), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 43) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

print(train.head())

'''
more than 50% of 1st class are from S embark  
more than 50% of 2nd class are from S embark  
more than 50% of 3rd class are from S embark

**fill out missing embark with S embark**
'''
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train.head())

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

print(train.head())

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 20, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[ dataset['Fare'] > 30, 'Fare'] = 2

print(train.head())


train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

cabin_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

print(train.head())


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

print(train.head())

features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape

print(train_data.head())

# Importing Classifier Modules
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# kNN Score
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("KNN: ",round(np.mean(score)*100, 2))

# decision tree Score
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("DT: ",round(np.mean(score)*100, 2))


# Ramdom Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("RF: ",round(np.mean(score)*100, 2))



# Naive Bayes Score
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("NB: ",round(np.mean(score)*100, 2))


# SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("SVC: ",round(np.mean(score)*100,2))


# CV + Random Forest
def RF_Best():
    estimator_grid = np.arange(15, 22, 1)
    depth_grid = np.arange(4, 6, 1)
    parameters = {'n_estimators': estimator_grid, 'max_depth': depth_grid}
    gridCV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=10)
    gridCV.fit(train_data, target)
    best_n_estim = gridCV.best_params_['n_estimators']
    best_depth = gridCV.best_params_['max_depth']
    print("best: ",best_depth, best_n_estim)

    RF_best = RandomForestClassifier(max_depth=best_depth,n_estimators=best_n_estim,random_state=3)
    # RF_best.fit(train_data, target)
    score = cross_val_score(RF_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("RF_best: ",round(np.mean(score)*100, 2))

    clf = RandomForestClassifier(max_depth=best_depth, n_estimators=best_n_estim, random_state=3)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()


# CV + AdaBoost
def AB_Best():
    estimator_grid = np.arange(50, 80, 2)
    learning_rate_grid = np.array([0.1,0.2,0.25,0.3,0.35,0.4,0.5])
    parameters = {'n_estimators': estimator_grid, 'learning_rate': learning_rate_grid}
    gridCV = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, cv=10)
    gridCV.fit(train_data, target)
    best_n_estim = gridCV.best_params_['n_estimators']
    best_learn_rate = gridCV.best_params_['learning_rate']
    print("Ada Boost best n estimator : " + str(best_n_estim))
    print("Ada Boost best learning rate : " + str(best_learn_rate))

    AB_best = AdaBoostClassifier(n_estimators=best_n_estim,learning_rate=best_learn_rate,random_state=3)
    AB_best.fit(train_data, target);
    score = cross_val_score(AB_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("AdaBoost: ",round(np.mean(score)*100, 2))

    clf = AdaBoostClassifier(n_estimators=best_n_estim, learning_rate=best_learn_rate, random_state=3)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()


# CV + SVM
def SVM_Best():
    C_grid = [0.001, 0.01, 0.3, 1, 3]
    gamma_grid = [0.001, 0.03, 0.1, 0.3]
    parameters = {'C': C_grid, 'gamma': gamma_grid}
    gridCV = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10);
    gridCV.fit(train_data, target)
    best_C = gridCV.best_params_['C']
    best_gamma = gridCV.best_params_['gamma']

    print("SVM best C : " + str(best_C))
    print("SVM best gamma : " + str(best_gamma))

    SVM_best = SVC(C=best_C, gamma=best_gamma)
    SVM_best.fit(train_data, target)
    score = cross_val_score(SVM_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("SVM best: ",round(np.mean(score)*100, 2))

    clf = SVC(C=best_C, gamma=best_gamma)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()

# result
def result():
    clf = SVC()
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
            "PassengerId": test["PassengerId"],
            "Survived": prediction
        })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()


RF_Best()
# AB_Best()
# SVM_Best()