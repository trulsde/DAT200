import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

train_df = pd.DataFrame(pd.read_csv("train.csv"))
test_df = pd.DataFrame(pd.read_csv("test.csv"))

columns_to_drop = ["Average Temperature During Storage (celcius)", "Harvest Time"]
train_df = train_df.drop(columns_to_drop, axis=1) # I'm dropping columns here, not rows!
test_df = test_df.drop(columns_to_drop, axis=1)

colour_mapping = {"red": 0, "green": 1, "yellow": 2}
train_df["color"] = train_df["color"].replace(colour_mapping)
test_df["color"] = test_df["color"].replace(colour_mapping)
print(train_df.dtypes)

def binary_mapping(value):
    return 1 if value > 0 else 0

y = train_df["Scoville Heat Units (SHU)"].values
y_binary = np.array([binary_mapping(value) for value in y]) # Total binary y for the very end
X = train_df.drop("Scoville Heat Units (SHU)", axis=1).values

train_df_gt0 = train_df[train_df["Scoville Heat Units (SHU)"] != 0]

y_gt0 = train_df_gt0["Scoville Heat Units (SHU)"].values
X_gt0 = train_df_gt0.drop("Scoville Heat Units (SHU)", axis=1).values

X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.7, random_state=42)
X_train_gt0, X_dev_gt0, y_train_gt0, y_dev_gt0 = train_test_split(X_gt0, y_gt0, train_size=0.7, random_state=42)

y_train_binary = np.array([binary_mapping(value) for value in y_train])
y_dev_binary = np.array([binary_mapping(value) for value in y_dev])

# Regression analysis
pls = Pipeline([
    ('imputer', SimpleImputer()),
    ('pls', PLSRegression())
])

# Simple linear regression
lr = Pipeline([
    ('imputer', SimpleImputer()),
    ('lr', LinearRegression(n_jobs=-1))
])

# Ensemble classifier
rf = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=.9)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
])

print(f"PLS parameters: {pls.get_params()}")
print(f"Randomforest parameters: {rf.get_params()}")

components = np.arange(1, 10)

estimators = np.arange(1, 1300)
depths = np.arange(1, 15)
learning_rates = np.linspace(0, 1, 10)

pls_grid = {"pls__n_components": components}

rf_grid = {"rf__n_estimators": estimators, "rf__max_depth": depths}

rf_search = RandomizedSearchCV(estimator=rf,
                                param_distributions=rf_grid,
                                scoring='f1',
                                cv=10,
                                n_jobs=-1)

pls_search = GridSearchCV(estimator=pls,
                          param_grid=pls_grid,
                          scoring='neg_mean_absolute_error',
                          cv=10,
                          n_jobs=-1,
                          verbose=2)

lr_search = GridSearchCV(estimator=lr,
                        param_grid={}, # The linear regression has no grid, but I sent it through the grid search for the sake of CV
                        scoring='neg_mean_absolute_error',
                        cv=10,
                        n_jobs=-1,
                        verbose=2)

# RandomForest:
rf_search.fit(X_train, y_train_binary)
print(f'Best parameters for RandomForest: {rf_search.best_params_}')
print(f'Best score for RandomForest: {abs(rf_search.best_score_)}')

# PLS for y > 0:
pls_search.fit(X_train_gt0, y_train_gt0)
print(f'Best parameters for PLS y > 0: {pls_search.best_params_}')
print(f'Best score for PLS: {abs(pls_search.best_score_)}')

# PLS else:
pls_search.fit(X_train, y_train)
print(f'Best parameters for PLS: {pls_search.best_params_}')
print(f'Best score for PLS: {abs(pls_search.best_score_)}')

# Linear Regression for y > 0:
lr_search.fit(X_train_gt0, y_train_gt0)
print(f'Best parameters for LinReg y > 0: {lr_search.best_params_}')
print(f'Best score for LinReg: {abs(lr_search.best_score_)}')

# Linear Regression else:
lr_search.fit(X_train, y_train)
print(f'Best parameters for LinReg: {lr_search.best_params_}')
print(f'Best score for LinReg: {abs(lr_search.best_score_)}')

# Finally, I choose linear regression as the linear model, since it acchieved a slightly higher score than PLS.
best_rf = rf_search.best_estimator_
lr = lr_search.best_estimator_

# Regular linear regression (Task A):
y_pred_linear = lr.predict(X_dev)

# Binary classification with RandomForest and linear classification with regular linear regression (Task B):
y_pred_ensemble = best_rf.predict(X_dev) # These are now binary
indices_where_1 = np.where(y_pred_ensemble == 1)[0]
cont_dev = X_dev[indices_where_1]
cont_pred = lr.predict(cont_dev)

y_pred_ensemble[indices_where_1] = cont_pred

print(f"Dev mean absolute error linear: {mean_absolute_error(y_dev, y_pred_linear)}")
print(f"Dev mean absolute error ensemble: {mean_absolute_error(y_dev, y_pred_ensemble)}")

best_rf.fit(X, y_binary)
X_test = test_df.values
y_pred = best_rf.predict(X_test)
indices_where_1 = np.where(y_pred == 1)[0]
cont_test = X_test[indices_where_1]
cont_pred = lr.predict(cont_test)

y_pred[indices_where_1] = cont_pred

submission_df = pd.DataFrame(y_pred, columns=['Scoville Heat Units (SHU)'])
submission_df.to_csv("Ensemble.csv", index_label="index")
