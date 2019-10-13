#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import tensorflow as tf
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import  StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC

#%%
df = pd.read_csv('data/creditcard.csv')
df.head()

#%%
# PCA features appear to be centered at 0 but aren't standardized with a std=0
# To create PCA, the features were scaled prior to the transformation
# The amount has a long tail.
df.describe()

#%%
# No NaNs in any column
df.isna().any()

#%%
# Fraud: 0.0017, not: 0.9983
print(f'Fraud: {df.Class.sum() / df.shape[0]:.4f}')
print(f'Not fraud: {1 - (df.Class.sum() / df.shape[0]):.4f}')

#%%
# Scale time and amount to be similar to the other features
s_scaler = StandardScaler()
r_scaler = RobustScaler()

# Robust to handle outliers, standard otherwise
amounts = df.Amount.values.reshape(-1,1)
times = df.Time.values.reshape(-1,1)
df.loc[:, 'scaled_amount'] = r_scaler.fit_transform(amounts)
df.loc[:, 'scaled_time'] = s_scaler.fit_transform(times)

df.drop(['Time', 'Amount'], axis=1, inplace=True)

#%%
# For cross validation, over/under sample on training data during CV to prevent
# leakage. Split data prior to sampling techniques
X = df.drop('Class', axis=1)
y = df.Class

skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
for train_idx, test_idx in skf.split(X, y):
    orig_X_train, orig_X_test = X.iloc[train_idx], X.iloc[test_idx]
    orig_y_train, orig_y_test = y.iloc[train_idx], y.iloc[test_idx]

orig_X_train = orig_X_train.values
orig_X_test = orig_X_test.values
orig_y_train = orig_y_train.values
orig_y_test = orig_y_test.values

# See if train and test label distributions are similarly distributed
train_labels, train_counts = np.unique(orig_y_train, return_counts=True)
test_labels, test_counts = np.unique(orig_y_test, return_counts=True)

print(train_counts / len(orig_y_train))
print(test_counts / len(orig_y_test))

#%%
# Make a new dataframe that contains the same amount of fraud / non in order to
# exam correlations between features
