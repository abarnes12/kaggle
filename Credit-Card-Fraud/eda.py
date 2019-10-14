# This file has been configured to run as a Jupyter notebook in Visual Studio Code

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
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, precision_recall_curve
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
# Random under sampling
def under_sample(df, rs=0):
    # make a copy and shuffle the rows
    df = df.copy().sample(frac=1)

    num_fraud = np.sum(df.Class)
    fraud = df.loc[df.Class == 1]
    non_fraud = df.loc[df.Class == 0][:num_fraud]

    even_df = pd.concat([fraud, non_fraud])

    # shuffle rows
    new_df = even_df.sample(frac=1, random_state=rs)

    return new_df

#%%
df_under = under_sample(df)
print(df_under.Class.value_counts()/len(df_under))
df_under.head()

#%% Correlation matrix
# First look at imbalanced data before looking at how the correlations change 
# with under sampling
sns.heatmap(df.corr()); plt.show()
sns.heatmap(df_under.corr()); plt.show()

#%%
# Plot the top features with a negative correlation larger then 0.5
neg_corr = df_under.corr()['Class'].sort_values()
neg_corr = neg_corr[neg_corr < -0.5]

for i in list(neg_corr.index):
    sns.boxplot(x='Class', y=i, data=df_under)
    plt.show()

#%%
# Plot the top features with a positive correlation larger then 0.5
pos_corr = df_under.corr()['Class'].sort_values()
pos_corr = pos_corr[pos_corr > 0.5]

for i in list(pos_corr.index):
    sns.boxplot(x='Class', y=i, data=df_under)
    plt.show()

#%%
def remove_extreme_outliers(df, cols, thresh):
    df = df.copy()
    for c in cols:
        series = df[c].loc[df['Class'] == 1].values
        q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
        iqr = q75 - q25
        cutoff = iqr * thresh
        lower = q25 - cutoff
        upper = q75 + cutoff

        #outliers = [x for x in series if x < lower or x > upper]
        to_drop = df[(df[c] > upper) | (df[c] < lower)].index
        df = df.drop(to_drop)
    return df

#%%
# Plot the distribution before and after removal of extreme outliers
# V14
v14 = df_under.V14[df_under.Class == 1].values
sns.distplot(v14); plt.show()

q25, q75 = np.percentile(v14, 25), np.percentile(v14, 75)
v14_iqr = q75 - q25

#%%
# Make a new dataframe that contains the same amount of fraud / non in order to
# exam correlations between features

#%%
# Create pipeline for creating training and test sets
# Perform under/over sampling
# Run LogisticRegression on the original data and each adjusted data set
# Compare the metrics (AUPRC)

#%%
def plot_auprc(clf, X_test, y_test):
    score = clf.decision_function(X_test)
    ap_score = average_precision_score(y_test, score)
    precision, recall, _ = precision_recall_curve(y_test, score)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'AP score: {ap_score}')
    plt.show()
