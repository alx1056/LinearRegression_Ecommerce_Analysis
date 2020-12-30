# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ___
# ## Title: Linear Regression of Ecommerce
# ### Author: Alex Fields
# 
# The purpose of this project is to walk-through and showcase how to run exploratory analysis and run linear regression on an Ecommerce dataset. 
# We will be trying to predict which is the most cost effective move for this company to make in terms of what is their best "KPI". 
# 
# The Ecommerce company is trying to decide whether to focus their efforts on their mobile app experience or their website. 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
customers = pd.read_csv("Ecommerce Customers")
customers.head()


# %%
customers.describe()


# %%
customers.info()


# %%
sns.jointplot(data=customers, x="Time on Website", y="Yearly Amount Spent", kind="reg")


# %%
sns.jointplot(data=customers, x="Time on App", y="Yearly Amount Spent", kind="reg")


# %%
sns.jointplot(data=customers, x="Time on App", y="Length of Membership", kind="hex")


# %%
sns.pairplot(data=customers)


# %%
sns.lmplot(data=customers, x="Length of Membership", y="Yearly Amount Spent")

# %% [markdown]
# Training and Testing Data

# %%
customers.columns


# %%
y = customers['Yearly Amount Spent']


# %%
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]


# %%
from sklearn.model_selection import train_test_split


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# %%
from sklearn.linear_model import LinearRegression


# %%
lm = LinearRegression()


# %%
lm.fit(X_train, y_train)


# %%
print(list(zip(X, lm.coef_)))


# %%
predictions = lm.predict(X_test)


# %%
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')


# %%
from sklearn import metrics


# %%
print("MAE of Model: ",metrics.mean_squared_error(y_test, predictions))


# %%
print("MSE of Model: ",metrics.mean_absolute_error(y_test, predictions))


# %%
print("RMSE of Model: ",np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# %%
#This gives a very accurate r^2 score to show high correlation and accuracy of model
print("R-Sqaured: ", round(metrics.explained_variance_score(y_test, predictions),4)*100,"%")


# %%
sns.distplot((y_test-predictions), bins=50)


# %%
pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])

# %% [markdown]
# If all predictors are held fixed, this will show: 
# for a 1 unit increase in avg session length with an average of $26$ dollars per year spent
# 
# for a 1 unit increase in Time on App with an average of $38$ dollars per year spent
# 
# for a 1 unit increase in Time on Website with an average of $0.30$ dollars per year spent
# 
# for a 1 unit increase in Length of Membership with an average of $61$ dollars per year spent
# 

# %%



