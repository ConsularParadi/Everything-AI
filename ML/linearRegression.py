#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%%
class LinearRegression:
    def __init__(self) -> None:
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            x = X_train[i] - np.mean(X_train)
            num += x*(y_train[i] - np.mean(y_train))
            den += x**2
        self.m = num/den
        self.b = np.mean(y_train) - self.m*np.mean(X_train)

    def predict(self, X_test):
        return self.m*X_test + self.b

#%%
df = pd.read_csv("../Datasets/Salary_dataset.csv")
df = df.drop(df.columns[0], axis=1)
df.head()

#%%
X = df.iloc[:,0].values
y = df.iloc[:,1].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)

# %%
lreg = LinearRegression()
lreg.fit(X_train, y_train)

# %%
lreg.predict(X_test[0])