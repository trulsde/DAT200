import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier


# Reading data

# I read the data in as a pandas dataframe, so that it can be managed with pandas features
data = pd.read_csv("train.csv")
df = pd.DataFrame(data)
df = df.drop(columns='Unnamed: 0')

# Data exploration and visualisation

# Plotting a heatmap of the linear correlation coefficient between all features in the dataset, to check for linear correlation
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.plot()

print(len(df))

"""Since there was little linear correlation between any two features, I plot scatter plots of all pairs to check for any other
kind of correlation"""

features = list(df)
target = 'Edible'

features.remove('Edible')
features1 = features.copy()
features2 = features.copy()

fig, ax = plt.subplots(20, 6, figsize=(20, 70))

row = 0
col = 0

for feat1 in features1:  
    for feat2 in features2:
        if feat2 == feat1:
            continue
        sns.scatterplot(x=feat1, y=feat2, hue=target, data=df, palette={0: 'coral', 1: 'lightskyblue'}, ax=ax[row][col])
        if col == 5:
            col = 0
            row += 1
            print(f'Row {row} complete')
        else:
            col += 1
    features2.remove(feat1)

plt.tight_layout()
plt.show()

"""

categories = [train_df.]
"""


# __Comments on the initial visualisation__:
# After pairing features and visualising them in scatter plots with target label `Edible`as hue parameter, it seems as if the pH feture, first and foremost, is going to be of great significance to the classification. This is because almost all the plots with pH as one of the features display more or less clearly separable data points. Secondly come the size features (circumference, length, weight...), which also seem to create tidy plots.

# One can also see some outliers, above all in the pH-values. This is something I didn't take into account in my calculations, because I thought it would harm the accuracy of the eventual model to get rid of these. This is a decision I grew uncertain of later on, as they may have made both the standardisation, the swapping of NaN-values with mean-values and the final predictions less accurate. 

# Data cleaning:

# X / y split:
y = pd.DataFrame(df['Edible'])
X = df.drop(y, axis=1)
y = y['Edible'].tolist()

"""As shown above, there are very many rows with NaN-values scattered over many different columns. I thus see removing rows 
and/or columns with NaN-values as inconvenient, considering the fairly small size of the dataset. I chose the mean-strategy instead,
which I thought would be a good substitute - although I see now, that I maybe should have taken care of the outliers first then, to get a cleaner mean."""


# Feature imputing:
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Don't want to remove any rows, because the train set is very small
imputer.fit(X)
X = imputer.transform(X)
X = pd.DataFrame(X, columns=['Acoustic Firmness Index', 'Atmospheric Pressure at Harvest (Pa)',
       'Bitterness Scale', 'Circumference (mm)', 'Color Intensity (a.u.)',
       'Find Distance from Main Vulcano (km)', 'Length (mm)',
       'Luminescence Intensity (a.u.)', 'Magnetic orientation (degree)',
       'Odor index (a.u.)', 'Seed Count', 'Skin Thickness (mm)',
       'Soil pH where Grown', 'Sugar Content (mg)', 'Weight (mg)', 'pH'])

# Mean and std computation:
mean = X.mean()
std = X.std()

# Train / test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # I chose a .25 share test set, because I thought it seemed like a good share
X_eval = pd.DataFrame(pd.read_csv("test.csv").drop('Unnamed: 0', axis=1))

# __Random Forest__:

# As many of the features seemed to show some kinds of non-linear correlation, I thought of the random forest and SVM classifiers, which can make non-linear decision boundaries. First out, I tried various hyperparameters for the random forest:

"""I decided to try random forest, because it's effective and deals well with non-linear decision boundaries. As seen in the initial
visualisation, the two classes are namely separated non-linearly. I also experimented with kNN before this, and Kernel SVM afterwards, but 
since it was strictly threatened with rejection for submissions that didn't contain only the MOST NECESSARY, I deleted the lot of it. Still think it was
an important part of the process though."""

with open('random_forest.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Estimators', 'Jobs', 'Depth', 'Criterion', 'Accuracy Train', 'Accuracy Test'])

# I created a loop that takes ages, but computes a thorough list over parameters and performance on the test set and stores it in a .csv-file
"""
    for estimators in range(1000, 1401, 100):
        print(f'{estimators} estimators:')
        for jobs in range(3, 7):
            print(f'{jobs} jobs:')
            for depth in range(2, 11, 2):
                print('Depth: ', depth)
                rf = RandomForestClassifier(n_estimators=estimators, random_state=42, n_jobs=jobs, max_depth=depth, criterion='gini')
                rf.fit(X_train, y_train)

                y_pred_t = rf.predict(X_train)
                accuracy_train = accuracy_score(y_pred_t, y_train)

                y_pred = rf.predict(X_test)
                accuracy_test = accuracy_score(y_pred, y_test)

                csv.writer(f).writerow([estimators, jobs, depth, accuracy_train, accuracy_test])
"""

# After looking through the resulting .csv-file, i found that 1200-1400 estimators, 3 jobs and a max depth of 10 were the best parameters.

# Testing feature selection on the best performing RandomForest:
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=10)

sfs = SequentialFeatureSelector(knn, n_features_to_select=5, direction="forward", n_jobs=4)
sfs = sfs.fit(X_train, y_train)

rf2 = RandomForestClassifier(n_estimators=1800, random_state=42, n_jobs=4, max_depth=15, criterion='gini')

X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)
X_eval_sfs = sfs.transform(X_eval)
X_sfs = sfs.transform(X)

rf2.fit(X_train_sfs, y_train)
y_pred = rf2.predict(X_test_sfs)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)

# It didn't work out well


# ### Kaggle submission

"""The .csv-list over random forest performances indicated that 1200-1400 estimators, 3 jobs and max depth 10 would do.
I then made this code to create .csv-lists of predictions for different random forests with nearby hyperparameters. Surprisingly,
the first one I uploaded - 1200 estimators, 3 jobs, 10 max depth - still holds the record amongst my submissions with .91452, although I did upload
several models with the parameters tweaked a bit to check for wiggle room."""

rf_final = RandomForestClassifier(n_estimators=1200, random_state=42, n_jobs=10, max_depth=200)

rf_final.fit(X, y)
y_pred = rf_final.predict(X_eval)

rf_df = pd.DataFrame(y_pred, columns=['Edible'])
rf_df.to_csv(f'RandomForest_1200_3_10.csv', index_label='index')

