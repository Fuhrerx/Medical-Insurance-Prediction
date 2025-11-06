# %% [markdown]
# ### **Importing Libraries/Frameworks**

# %%
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## **Reading CSV and Inspecting**

# %%
df = pd.read_csv(r"C:\Portfoilo Projects\Medical-Insurance-Prediction\Data\insurance.csv")
df

# %%
# df.info()
# %%
# df.describe()

# %%
df_1 = df.drop(columns= 'sex', axis = 1)
df_1 = df.drop(columns= 'region', axis = 1)
df_1 = df.drop(columns= 'smoker', axis = 1)

# %% [markdown]
# ## **Plotting**

# %%
# df_1.hist('charges')
# plt.show

# %%
df_1['log_charges'] = np.log2(df_1['charges'])
df['log_charges'] = np.log2(df['charges'])
df_1['is_smoker'] = df['smoker'] == 'yes'

# df_1.hist('log_charges')
# plt.show

# %%
df.drop('smoker', axis =1, inplace = True)
df.drop('region', axis =1, inplace = True)
df.drop('sex', axis =1, inplace = True)
corr = df.corr()
corr

# %%
# sns.heatmap(corr, cmap='Blues', annot=True)
# plt.show

# %%
df_numrc = df_1[['age', 'bmi', 'children', 'charges', 'log_charges']]
# sns.pairplot(df_numrc, kind='scatter', plot_kws={'alpha': 0.4})

# %% [markdown]
# ## **Model 1 Building**

# %%
# X = df_1[['age', 'bmi', 'is_smoker']]
# y = df_1['log_charges']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# # %%
# X_train

# # %%
# X_test

# # %%
# y_train

# %%
# y_test

# # %%
# model = LinearRegression()
# model.fit(X_train, y_train)
# model.coef_

# # %%
# y_pred = model.predict(X_train)

# # %%
# y_pred

# # %%
# train_mse = mean_squared_error(y_train, y_pred)
# train_mse

# # %%
# #undo log transformation
# train_mse_org = np.exp2(mean_squared_error(y_train, y_pred))
# train_mse_org

# # %%
# train_r2 = r2_score(y_train, y_pred)
# train_r2

# # %% [markdown]
# # ## **Model 1 Evaluation**

# # %%
# plot_df = pd.DataFrame({
#     'Predictions' : y_pred,
#     'Actual' : X_train['is_smoker'],
#     'age' : X_train['age'],
#     'bmi' : X_train['bmi'],
#     'Residuals' : y_train - y_pred,
# })

# %%
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Predictions', y='Actual', data=plot_df, alpha=0.7)
# plt.xlabel('Predicted log_charges')
# plt.ylabel('Acual log_charges')

# plt.show()

# %%
# sns.scatterplot(x='Predictions', y='Residuals', data=plot_df, alpha=0.7)

# %%
# test_pred = model.predict(X_test)

# mean_squared_error(y_test, test_pred)

# # %%
# np.exp2(mean_squared_error(y_test, test_pred))

# %% [markdown]
# ## **Model 2 Building**

# %%
smokers_df =df_1[df_1['is_smoker'] == True]

# %%
X = smokers_df[['age', 'bmi']]
y = smokers_df['log_charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
smoker_model = LinearRegression(fit_intercept= True)
smoker_model.fit(X_train, y_train)
smoker_model.coef_

# %%
y_pred = smoker_model.predict(X_train)

# %% [markdown]
# ## **Model 2 Evaluation**

# %%
train_mse = mean_squared_error(y_train, y_pred)
train_mse

# %%
train_r2 = r2_score(y_train, y_pred)
train_r2

# %%
# plot_df = pd.DataFrame({
#     'Predictions' : y_pred,
#     'Actual' : y_train,
#     'age' : X_train['age'],
#     'bmi' : X_train['bmi'],
#     'Residuals' : y_train - y_pred,
# })
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Predictions', y='Actual', data=plot_df, alpha=0.7)
# plt.plot([13.5, 16], [13.5, 16], 'k--', alpha = 0.5)
# plt.xlabel('Predicted log_charges')
# plt.ylabel('Acual log_charges')

#plt.show()

# %%
# sns.scatterplot(x='Predictions', y='Residuals', data=plot_df, alpha=0.7)

# %%
test_pred = smoker_model.predict(X_test)

mean_squared_error(y_test, test_pred)

# %% [markdown]
# ## **Hyperparameter Tuning**

# %%
# Define the model
# model = Ridge()

# # Define the hyperparameter grid
# param_grid = {
# 'alpha': [0.1, 1.0, 10.0, 100.0], # Regularization strength
# 'fit_intercept': [True, False]
# }

# # Perform grid search
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Output the best parameters and score
# print("Best Hyperparameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# %% [markdown]
# ## **Predicting With Made Up Data**

# %%

class Predictor :

    def __init__(self, age, bmi):
        self.age = age
        self.bmi = bmi
        model_test = pd.DataFrame([self.age, self.bmi]).transpose()
        model_test.columns = ['age', 'bmi']
        self.new_pred = smoker_model.predict(model_test)
        self.new_pred = (np.exp2(self.new_pred))

    def __str__(self):
        return f"{float(self.new_pred.round(3)):,.2f}"
