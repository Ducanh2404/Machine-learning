#importing important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#reading the dataset
df = pd.read_csv("Data.csv")

# drop nan values
df.dropna(inplace=True)

# instantiate labelencoder object
le = LabelEncoder()
# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# Get list of categorical column names
categorical_cols = df.columns[categorical_feature_mask].tolist()
# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

#split dataset into train and test
train, test = train_test_split(df, test_size=0.3, random_state=0)

x_train = train.drop('world',axis=1)
y_train = train['world']

x_test = test.drop('world',axis=1)
y_test = test['world']

model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(x_train, y_train)
accuracy = model.score(x_test,y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))