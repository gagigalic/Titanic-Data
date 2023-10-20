import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("titanic_train.csv")
print(train.head())
train.info()

#false not null, true null
print(train.isnull())

# to see where is empty (with boolean values)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.savefig("heatmap.png")
plt.close()

#to see who survived, 0 not, 1 yes
sns.set_style("whitegrid")
sns.countplot(x="Survived", data = train)
plt.savefig("survived.png")
plt.close()

#survived with male/female
sns.countplot(x="Survived", hue = "Sex", data = train, palette = "RdBu_r")
plt.savefig("survived2.png")
plt.close()

#passanger class
sns.countplot(x="Survived", hue = "Pclass", data = train)
plt.savefig("survived3.png")
plt.close()

sns.displot(train["Age"].dropna(), kde = False, bins = 30)
plt.savefig("displot.png")
plt.close()

train["Age"].plot.hist(bins=35)

sns.countplot(x="SibSp", data = train)
plt.savefig("SibSp.png")
plt.close()

#price
train["Fare"].hist(bins = 40, figsize=(10,4))
plt.savefig("Fare.png")
plt.close()

#Pclass vs Age
sns.boxplot(x="Pclass", y="Age", data=train)
plt.savefig("Pclass.png")
plt.close()

#replace empty age to aveger age for Pclass

def impute_age(cols):
    Age = cols[0]
    Pclass= cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return  37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar =False)
plt.savefig("heatmap2.png")
plt.close()

#remove Cabin becouse have much empty values
train.drop("Cabin", axis=1, inplace=True)

sns.heatmap(train.isnull(), yticklabels=False, cbar =False)
plt.savefig("heatmap3.png")
plt.close()

# clear one or more misiing values
train.dropna(inplace=True)

#dummy variable
sex =pd.get_dummies(train["Sex"], drop_first=True)
embark = pd.get_dummies(train["Embarked"], drop_first=True)
train = pd.concat([train, sex, embark], axis = 1)

train.drop(["Sex", "Embarked", "Name", "Ticket"], axis = 1, inplace = True)
train.drop(["PassengerId"], axis = 1, inplace = True)
train.head()


X = train.drop('Survived',axis=1)
y = train["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))