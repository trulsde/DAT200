import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Usual organising in data frames and overview of column names
train = pd.read_csv("train.csv")
train_df = pd.DataFrame(train).drop(["Unnamed: 0", "index"], axis=1)

test = pd.read_csv("test.csv")
test_df = pd.DataFrame(test).drop(["Unnamed: 0", "index"], axis=1)

diagnoses = train_df["Diagnosis"].unique().tolist()

# train-test-split
y = train_df['Diagnosis'].values
X = train_df.drop('Diagnosis', axis=1).values
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, random_state=42)

"""
I decided to shove the dataset with one-hot encoded categorical columns through the scalers of the pipelines, although this is not ideal. But this
should not be of any significance in this case, as we're only interested in the predictive results.
"""

rf = Pipeline([
     ('scaler', StandardScaler()),
     ('pca', PCA()),
     ('rf', RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42))
])

svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(random_state=42))
])

knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

lgr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('lgr', LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced', random_state=42))
])

# Thought I'd go for a logistic regression model using the softmax vector classification too, since we're dealing with multi class classification
# (I study NLT, language technology at UiO too, where we were introduced to the softmax):
lgr_soft = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('lgr', LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced', random_state=42))
])

n_estimators_range = [200, 300, 400]
depth_range = [5, 10, 15, 20]

neighbor_range = [7, 8, 9, 10, 11, 12, 15, 20]

n_components = [4, 5, 6, 7, 8, 9]

gamma_range = [0.001, 0.01, 0.1, 1.0, 10, 100]
C_range = [100, 500, 1000]
degrees = [1, 2, 3]

rf_grid = {'rf__n_estimators': n_estimators_range, 'rf__max_depth': depth_range}

knn_grid = {'knn__n_neighbors': neighbor_range}

svm_grid = [{'svm__C': C_range, 'svm__kernel': ['linear'], 'pca__n_components': n_components},
           {'svm__C': C_range, 'svm__kernel': ['rbf'], 'svm__gamma': gamma_range, 'pca__n_components': n_components}]

lgr_grid = {'lgr__C': gamma_range, 'lgr__penalty': ['l2'], 'pca__n_components': n_components}

# creating grid search objects for all respective model pipelines:
rf_search = GridSearchCV(estimator=rf,
                           param_grid=rf_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                            verbose=1)

knn_search = GridSearchCV(estimator=knn,
                           param_grid=knn_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

svm_search = GridSearchCV(estimator=svm,
                            param_grid=svm_grid,
                            scoring='f1_macro',
                            cv=10,
                            n_jobs=-1,
                         verbose=2)

lgr_search = GridSearchCV(estimator=lgr,
                           param_grid=lgr_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

lgr_soft_search = GridSearchCV(estimator=lgr_soft,
                           param_grid=lgr_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

# Round 2 of search with the most successful models from previous search:
n_estimators_range2 = [500, 600, 700]
depth_range_2 = [3, 4, 6, 7]

components = [9, 10, 11, 12, 13]

gamma_range = [0.01, 0.1, 1.0, 10, 100, 1000]
C_range2 = [90, 100, 110]

rf_grid = {'rf__n_estimators': n_estimators_range, 'rf__max_depth': depth_range}

lgr_grid = {'lgr__C': gamma_range, 'lgr__penalty': ['l2'], 'pca__n_components': components}

svm_grid = {'svm__C': C_range, 'svm__kernel': ['rbf'], 'svm__gamma': gamma_range, 'pca__n_components': n_components}

rf_search = GridSearchCV(estimator=rf,
                           param_grid=rf_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                            verbose=1)

lgr_search = GridSearchCV(estimator=lgr_soft,
                           param_grid=lgr_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

lgr_soft_search = GridSearchCV(estimator=lgr_soft,
                           param_grid=lgr_grid,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

# K-fold-algorithm was copied from the lecture example, before variables like 'SEED' were altered:
scores_rf = []
scores_svm = []
scores_lgr = []
scores_lgr_soft = []

skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X_train, y_train)

rf_best = rf_search.best_estimator_
svm_best = svm_search.best_estimator_
lgr_best = lgr_search.best_estimator_
lgrs_best = lgr_soft_search.best_estimator_

for k, (train_idxs, validation_idxs) in enumerate(skfold):
    rf_best.fit(X_train[train_idxs], y_train[train_idxs])
    svm_best.fit(X_train[train_idxs], y_train[train_idxs])
    lgr_best.fit(X_train[train_idxs], y_train[train_idxs])
    lgrs_best.fit(X_train[train_idxs], y_train[train_idxs])

    y_pred_rf = rf_best.predict(X_train[validation_idxs])
    y_pred_svm = svm_best.predict(X_train[validation_idxs])
    y_pred_lgr = lgr_best.predict(X_train[validation_idxs])
    y_pred_lgrs = lgrs_best.predict(X_train[validation_idxs])

    rf_score = f1_score(y_train[validation_idxs], y_pred_rf, average='macro')
    svm_score = f1_score(y_train[validation_idxs], y_pred_svm, average='macro')
    lgr_score = f1_score(y_train[validation_idxs], y_pred_lgr, average='macro')
    lgrs_score = f1_score(y_train[validation_idxs], y_pred_lgrs, average='macro')

    scores_rf.append(rf_score)
    scores_svm.append(svm_score)
    scores_lgr.append(lgr_score)
    scores_lgr_soft.append(lgrs_score)

    print(
        f'Fold: {k + 1:2d}, Class dist.: {np.bincount(y_train[train_idxs])}, RF-score: {rf_score:.3f}, SVM-score: {svm_score:.3f}, LogReg-score: {lgr_score:.3f}, Softmax-score: {lgrs_score:.3f}')

"""
Kaggle upload; I train all three suitable models on the whole training set from before. The LogReg turned out to be the best model,
but I just make files for the other two as well, to be able to upload their results too.
"""

rf_best.fit(X, y)
svm_best.fit(X, y)
lgr_best.fit(X, y)

X_test = test_df.values

model_name_list = ["Randomforest", "SVM", "LogReg"]
count = 0
for model in [rf_best, svm_best, lgr_best]:
    y_pred = model.predict(X_test)
    # Using the previous 'diagnoses'-list (train_df['Diagnosis'].unique.tolist())
    y_pred = [diagnoses[entry] for entry in y_pred]

    submission_df = pd.DataFrame(y_pred, columns=['Diagnosis'])
    submission_df.to_csv(f'{model_name_list[count]}.csv', index_label='index')
    count += 1

