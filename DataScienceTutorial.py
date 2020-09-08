# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
# pretty print only the last output of the cell
InteractiveShell.ast_node_interactivity = "last_expr"
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3info.txt
data = pd.read_csv('data\data.csv')


# %%
print("Preprocessing Data")
col_names = data.columns.tolist()
print("Column Names:")
print(col_names)


# %%
print('\n Sample Data')
data[col_names].head(6)


# %%
print(data.shape)
print(data.dtypes)
# %%
for i in data.columns.tolist():
    k = sum(pd.isnull(data[i]))
    print(i, k)

# %%
data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})
# %%
for i in data.columns.tolist():
    k = sum(pd.isnull(data[i]))
    print(i, k)
# %%
print(data.describe(include=['int64', 'float64']))
print(data.describe(include=['object']))
# %%
print(data['survived'].value_counts())
# %%
print(data.dtypes)
# %%

fig, axs = plt.subplots(ncols=5, figsize=(30, 5))
sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])
# %%
data.replace({'male': 1, 'female': '0'}, inplace=True)
# %%
data[col_names].head(6)
# %%
data.corr().abs()[["survived"]]
# %%
data['relatives'] = data.apply(lambda row: int(
    (row['sibsp'] + row['parch']) > 0), axis=1)
data.corr().abs()[["survived"]]
# %%
data = data[['sex', 'pclass', 'age', 'relatives', 'fare', 'survived']].dropna()
# %%
print("Modeling Data")
x_train, x_test, y_train, y_test = train_test_split(
    data[['sex', 'pclass', 'age', 'relatives', 'fare']], data.survived, test_size=0.2, random_state=0)
# %%
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
# %%
model = GaussianNB()
model.fit(X_train, y_train)
# %%
predict_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict_test))
# %%

model = Sequential()
# %%
model.add(Dense(5, kernel_initializer='uniform',
                activation='relu', input_dim=5))
model.add(Dense(5, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# %%
model.summary()
# %%
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=32, epochs=50)
# %%
y_pred = model.predict_classes(X_test)
print(metrics.accuracy_score(y_test, y_pred))
# %%
InteractiveShell.ast_node_interactivity = "all"