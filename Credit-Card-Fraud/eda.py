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
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
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
def plot_auprc(score, ap_score, X_test, y_test):
    precision, recall, _ = precision_recall_curve(y_test, score)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'AP score: {ap_score}')
    plt.show()
    return

#%%
def prepare_data(df, scalers):
    df = df.copy() # don't alter the original
    # if scalers = None then use StandardScaler for all columns
    # add check for scalers = None
    for k, v in scalers.items():
        series = df[k].values.reshape(-1,1)
        df.loc[:, f'scaled_{k}'] = scalers[k].fit_transform(series)
        df.drop([k], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df.Class

    return (X, y)

def calc_metrics(clf, X_test, y_test):
    # calculate precision, recall, f1, accuracy
    # return dictionary
    y_pred = clf.predict(X_test)
    if hasattr(clf, 'decision_function'):
        score = clf.decision_function(X_test)
    else:
        score = clf.predict_proba(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, score)
    metrics = {'precision': precision,
               'recall': recall,
               'f1': f1,
               'accuracy': accuracy,
               'ap': ap,
               'score': score}
    return metrics

def run_raw_analysis(X_train, y_train, X_test, y_test, params, n_iter):
    #rand_lr = RandomizedSearchCV(LogisticRegression(solver='liblinear'),
    #                             params, 
    #                             n_iter=n_iter,
    #                             cv=3)
    #rand_lr.fit(X_train, y_train)
    #lr = rand_lr.best_estimator_
    lr = LogisticRegression(solver='liblinear').set_params(**params)
    lr.fit(X_train, y_train)
    metrics = calc_metrics(lr, X_test, y_test)

    return metrics

def run_under_analysis(X_train, y_train, X_test, y_test, params, n_iter):
    #rand_lr = RandomizedSearchCV(LogisticRegression(solver='liblinear'),
    #                             params, 
    #                             n_iter=n_iter,
    #                             cv=3)
    near_miss = NearMiss(sampling_strategy='majority')
    #under_pipe = imb_make_pipeline(near_miss, rand_lr)
    #under_model = under_pipe.fit(X_train, y_train)
    #lr = rand_lr.best_estimator_
    lr = LogisticRegression(solver='liblinear').set_params(**params)
    under_pipe = imb_make_pipeline(near_miss, lr)
    under_model = under_pipe.fit(X_train, y_train)
    metrics = calc_metrics(lr, X_test, y_test)

    return metrics

def run_over_analysis(X_train, y_train, X_test, y_test, params, n_iter):
    #rand_lr = RandomizedSearchCV(LogisticRegression(solver='liblinear'),
    #                             params, 
    #                             n_iter=n_iter,
    #                             cv=3)
    smote = SMOTE(sampling_strategy='minority')
    #over_pipe = imb_make_pipeline(smote, rand_lr)
    #over_model = over_pipe.fit(X_train, y_train)
    #lr = rand_lr.best_estimator_
    lr = LogisticRegression(solver='liblinear').set_params(**params)
    over_pipe = imb_make_pipeline(smote, lr)
    over_model = over_pipe.fit(X_train, y_train)
    metrics = calc_metrics(lr, X_test, y_test)

    return metrics

#%%
# read in data
df = pd.read_csv('data/creditcard.csv')

# set parameters
n_iter = 10
lr_params = {'penalty': ['l1', 'l2'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Robust to handle outliers, standard otherwise
scalers = {'Amount': RobustScaler(),
           'Time': StandardScaler()}
X, y = prepare_data(df, scalers)
X = X.drop('scaled_Time', axis=1)

# RandomSearchCV on full data to get rough parameters
rand_lr = RandomizedSearchCV(LogisticRegression(solver='liblinear'),
                             lr_params,
                             n_iter=n_iter,
                             cv=3)
rand_lr.fit(X, y)
best_params = rand_lr.best_params_

# SKF to create train test
# Loop over another SKF to split train into train val

#%%
skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=False)
raws = []
unders = []
overs = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    print('Raw')
    metrics_raw = run_raw_analysis(X_train, y_train,
                                   X_test, y_test,
                                   best_params, n_iter)
                                   #lr_params, n_iter)
    print(metrics_raw)
    raws.append(metrics_raw['ap'])
    plot_auprc(metrics_raw['score'], metrics_raw['ap'], X_test, y_test)

    print('Under')
    metrics_under = run_under_analysis(X_train, y_train,
                                       X_test, y_test,
                                       best_params, n_iter)
                                       #lr_params, n_iter)
    print(metrics_under)
    unders.append(metrics_under['ap'])
    plot_auprc(metrics_under['score'], metrics_under['ap'], X_test, y_test)

    print('Over')
    metrics_over = run_over_analysis(X_train, y_train,
                                     X_test, y_test, 
                                     best_params, n_iter)
                                     #lr_params, n_iter)
    print(metrics_over)
    overs.append(metrics_over['ap'])
    plot_auprc(metrics_over['score'], metrics_over['ap'], X_test, y_test)


#%%
print(f'Raw avg: {np.mean(raws)}')
print(f'Under avg: {np.mean(unders)}')
print(f'Over avg: {np.mean(overs)}')

#%%
# The plots from the train/test loop shows that undersampling does not perform
# well at all. Interestingly enough, the oversampled average is about the same
# as the raw data. There must be more to do

# Remove extreme outliers for under- and over-sampled data
# Is Time useful? It's the time from the first transaction. Since the other
# features are numerical, I don't see how they can contain info like "ID 1 
# spent X at this time and Y at this time which are too far apart."