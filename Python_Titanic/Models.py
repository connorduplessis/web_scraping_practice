#Titanic Kaggle Submissions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

#Read in Data
train_df = pd.read_csv('/Users/connorduplessis/Documents/Python/Python_Titanic/train.csv')
test_df = pd.read_csv('/Users/connorduplessis/Documents/Python/Python_Titanic/test.csv')
combine = pd.concat([train_df, test_df], sort=True)
test_df
#CLEANING/FEATURE ENGINEERING
#calculate sum and proportion of NaN values for each column
#we see 77% of the cabin variable is missing, along with about 20% of Ages
combine.isna().sum()
combine.isna().sum()/len(combine.index)


#remove cabin and ticket variables as they will not provide any useful info
combine = combine.drop(['Cabin', 'Ticket'], axis = 1)

#creating title variable from Name
#train set
names = combine['Name']
titles = names.str.split(',|\.', n = 2, expand = True)
unique_titles = titles[1].unique()
unique_titles

#rename columns
titles = titles.rename(index=str, columns={0: "Last Name", 1: "Title", 2: "First Name"})
titles

#add title, first, last name colomns into DF
combine = combine.drop('Name', axis = 1)
titles = titles.set_index(combine.index)
combine = pd.concat([combine, titles], axis = 1)

#convert rare titles
for row in combine:
    combine['Title'] = combine['Title'].replace(['Lady', 'the Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', regex = True)

    combine['Title'] = combine['Title'].str.replace('Mlle', 'Miss')
    combine['Title'] = combine['Title'].str.replace('Ms', 'Miss')
    combine['Title'] = combine['Title'].str.replace('Mme', 'Mrs')

combine['Title'].unique()
#locate Rarea title
combine.loc[combine['Title'] == ' Rarea']
#correct Rarea to Rare
combine['Title'] = combine['Title'].str.replace('Rarea', 'Rare')
combine['Title'].unique()

#Create family size variable from Sibsp and Parch variables
for row in combine:
    combine['Familysize'] = combine['SibSp'] + combine['Parch']

#Drop Sibsp and Parch in favor of Familysize
combine = combine.drop(['SibSp', 'Parch'], axis = 1)

#check na's again
combine.isna().sum()
#find most common Embarked and fill in for missing
combine['Embarked'].mode()
for row in combine:
    combine['Embarked'] = combine['Embarked'].fillna('S')

#replace missing fare with median fare for pclass
combine.loc[combine['Fare'].isna()]
combine[['Fare', 'Pclass']].groupby(['Pclass'], as_index= False).median()
combine['Fare'] = combine['Fare'].fillna(8.05)

#Replace missing Age with median age
combine['Age'].median()
for row in combine:
    combine['Age'] = combine['Age'].fillna(28)



#Analyzing correlation of factors
combine[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
combine[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
combine[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
combine[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
combine[['Familysize', 'Survived']].groupby(['Familysize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#Coverting all Variables to numerical for machine learning algorithms

#convert Male/Female to 0 & 1
for row in combine:
    combine['Sex'] = combine['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#convert Embarked to numeric
for row in combine:
    combine['Embarked'] = combine['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#convert title to numeric
combine['Title'] = combine['Title'].str.strip()

for row in combine:
    combine['Title'] = combine['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4} ).astype(int)

#drop name column
combine = combine.drop(['Last Name', 'First Name'], axis = 1)

#break back into separate df's for models
test_df2 = combine[combine['Survived'].isna()]
train_df2 = combine[combine['Survived'].isna() == False]


#models
X_train = train_df2.drop('Survived', axis = 1)
Y_train = train_df2["Survived"]
X_test  = test_df2.drop("Survived", axis=1).copy()


#GLM
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
GLM_pred = logreg.predict(X_test).astype(int)

GLM_submission = pd.DataFrame({
        "PassengerId": test_df2["PassengerId"],
        "Survived": GLM_pred
        })

GLM_submission.to_csv('GLM_submission.csv', index=False)


#random forest
rf = RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
              }

acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(rf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, Y_train)
rf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rf.fit(X_train, Y_train)

rf_predict = rf.predict(X_test).astype(int)

rf_submission = pd.DataFrame({
        "PassengerId": test_df2["PassengerId"],
        "Survived": rf_pred
        })

rf_submission.to_csv('rf_submission.csv', index = False)
