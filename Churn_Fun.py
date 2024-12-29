# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from churn_measurements import calibration, discrimination
from scipy.stats import mstats
import bisect
from scipy import interp
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pylab as plt
import numpy as np
from IPython import get_ipython

# %%
# File Name: Churn_Fun.ipynb


# %%
# load modules

import pandas as pd
pd.set_option('precision', 3)
get_ipython().run_line_magic('matplotlib', 'inline')
# pretty print only the last output of the cell
InteractiveShell.ast_node_interactivity = "last_expr"
warnings.filterwarnings('ignore')
# Load in the r magic
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
# We need ggplot2
get_ipython().run_line_magic('R', 'require(ggplot2)')
# local modules


# %%
print("~ Importing Data ~")
df = pd.read_csv("data\\churn.csv")


# %%
print("~ Preprocessing Data ~")
col_names = df.columns.tolist()
print("Column names:")
print(col_names)


# %%
to_show = col_names[:6] + col_names[-6:]
print('\n Sample data:')
df[to_show].head(6)


# %%
print(df.shape)
print(df.dtypes)


# %%
# The number of numeric data
len(df.select_dtypes(include=['int64', 'float64']).columns)


# %%
# The number of categorical data
len(df.select_dtypes(include=['category', 'object']).columns)


# %%
# Check there is any missing data
for i in df.columns.tolist():
    k = sum(pd.isnull(df[i]))
    print((i, k))


# %%
print("~ Exploring Data ~")
# numeric data
# print df.describe()
# same as above
print(df.describe(include=['int64', 'float64']))


# %%
# categorical and object data
print(df.describe(include=['category', 'object']))


# %%
print(df['Churn?'].value_counts())


# %%
print(df.groupby(df['Churn?']).mean())


# %%
# Histogram
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()


# %%
df.plot(kind='density', subplots=True, layout=(5, 4), sharex=False,
        legend=False, fontsize=1)
plt.show()


# %%
# Boxplots
df.plot(kind='box', subplots=True, layout=(5, 4), sharex=False,
        sharey=False, fontsize=1)
plt.show()


# %%
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, len(df._get_numeric_data().columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df._get_numeric_data().columns.tolist())
ax.set_yticklabels(df._get_numeric_data().columns.tolist())
plt.show()


# %%
print("~ Preparing Target and Features Data ~")

# Isolate target data
y = np.where(df['Churn?'] == 'True.', 1, 0)

# We don't need these columns
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
df = df.drop(to_drop, axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
# yes_no_cols will be re-used for later scoring
yes_no_cols = ["Int'l Plan", "VMail Plan"]
df[yes_no_cols] = df[yes_no_cols] == 'yes'

# Pull out fesatures for later scoring
features = df.columns

# feature variables
X = df.to_numpy().astype(np.float)


# %%
# Importances of features
train_index, test_index = train_test_split(df.index, random_state=4)

forest = RF()
forest_fit = forest.fit(X[train_index], y[train_index])
forest_predictions = forest_fit.predict(X[test_index])


importances = forest_fit.feature_importances_[:10]
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print(("%d. %s (%f)" % (f + 1, df.columns[f], importances[indices[f]])))

# Plot the feature importances of the forest
#import pylab as pl
plt.figure()
plt.title("Feature importances")
plt.bar(list(range(10)), importances[indices],
        yerr=std[indices], color="r", align="center")
plt.xticks(list(range(10)), indices)
plt.xlim([-1, 10])
plt.show()


# %%
print("~ Transforming Data ~")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels:", np.unique(y))


# %%
print("~ Building K-Fold Cross-Validations ~")


def run_cv(X, y, clf):
    # construct a K-Fold object
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    y_pred = y.copy()

    # iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


# %%
print("~ Evaluating Models ~")


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


print("Logistic Regression:")
print("Accuracy = %.3f" % accuracy(y, run_cv(X, y, LR())))
print("Gradient Boosting Classifier:")
print("Accuracy = %.3f" % accuracy(y, run_cv(X, y, GBC())))
print("Support vector machines:")
print("Accuracy = %.3f" % accuracy(y, run_cv(X, y, SVC())))
print("Random forest:")
print("Accuracy = %.3f" % accuracy(y, run_cv(X, y, RF())))
print("K-nearest-neighbors:")
print("Accuracy = %.3f" % accuracy(y, run_cv(X, y, KNN())))


# %%
# F1-Scores and Confusion Matrices
def draw_confusion_matrices(confusion_matricies, class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


y = np.array(y)
class_names = np.unique(y)


confusion_matrices = {
    1: {
        'matrix': confusion_matrix(y, run_cv(X, y, LR())),
        'title': 'Logistic Regression',
    },
    2: {
        'matrix': confusion_matrix(y, run_cv(X, y, GBC())),
        'title': 'Gradient Boosting Classifier',
    },
    3: {
        'matrix': confusion_matrix(y, run_cv(X, y, SVC())),
        'title': 'Support Vector Machine',
    },
    4: {
        'matrix': confusion_matrix(y, run_cv(X, y, RF())),
        'title': 'Random Forest',
    },
    5: {
        'matrix': confusion_matrix(y, run_cv(X, y, KNN())),
        'title': 'K Nearest Neighbors',
    },
}


fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in list(confusion_matrices.items()):
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii)  # starts from 1
    plt.title(title)
    sns.heatmap(matrix, annot=True,  fmt='')

print("Logisitic Regression F1 Score", f1_score(y, run_cv(X, y, LR())))
print("Gradient Boosting Classifier F1 Score",
      f1_score(y, run_cv(X, y, GBC())))
print("Support Vector Machines F1 Score", f1_score(y, run_cv(X, y, SVC())))
print("Random Forest F1 Score", f1_score(y, run_cv(X, y, RF())))
print("K-Nearest-Neighbors F1 Score", f1_score(y, run_cv(X, y, KNN())))


# %%
# ROC plots
def plot_roc(X, y, clf_class):
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
#    all_tpr = []
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' %
                 (i, roc_auc))
        i = i + 1
    mean_tpr /= kf.get_n_splits(X)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


print("Logisitic Regression:")
plot_roc(X, y, LR())

print("Gradient Boosting Classifier:")
plot_roc(X, y, GBC())

print("Support vector machines:")
plot_roc(X, y, SVC(probability=True))

print("Random forests:")
plot_roc(X, y, RF(n_estimators=18))

print("K-nearest-neighbors:")
plot_roc(X, y, KNN())


# %%
print("~ Building K-Fold Cross-Validations with Probabilities ~")


def run_prob_cv(X, y, clf):
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    y_prob = np.zeros((len(y), 2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


# %%
print("~ Calculating Calibration and Discrimination ~")
# Take on RF
pred_prob = run_prob_cv(X, y, RF(n_estimators=10, random_state=4))

# Use 10 estimators so predictions are all multiples of 0.1
pred_churn = pred_prob[:, 1].round(1)
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)
counts[:]


# %%
# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

counts = pd.concat([counts, true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts.sort_values('pred_prob').reset_index().drop(['index'], axis=1)


# %%
baseline = np.mean(is_churn)


# %%
get_ipython().run_cell_magic('R', '-i counts,baseline -w 800 -h 600 -u px',
                             '\nggplot(counts,aes(x=pred_prob,y=true_prob,size=count)) + \n    geom_point(color=\'blue\') + \n    stat_function(fun = function(x){x}, color=\'red\') + \n    stat_function(fun = function(x){baseline}, color=\'green\') + \n    xlim(-0.05,  1.05) + ylim(-0.05,1.05) + \n    ggtitle("RF") + \n    xlab("Predicted probability") + ylab("Relative  of outcome")')


# %%
def print_measurements(pred_prob):
    churn_prob, is_churn = pred_prob[:, 1], y == 1
    print("  %-20s %.4f" %
          ("Calibration Error", calibration(churn_prob, is_churn)))
    print("  %-20s %.4f" %
          ("Discrimination", discrimination(churn_prob, is_churn)))


# %%
print("Note -- Lower calibration is better, higher discrimination is better")
print("Logistic Regression:")
print_measurements(run_prob_cv(X, y, LR()))

print("Gradient Boosting Classifier:")
print_measurements(run_prob_cv(X, y, GBC()))

print("Support vector machines:")
print_measurements(run_prob_cv(X, y, SVC(probability=True)))

print("Random forests:")
print_measurements(run_prob_cv(X, y, RF(n_estimators=18)))

print("K-nearest-neighbors:")
print_measurements(run_prob_cv(X, y, KNN()))


# %%
print('~ Profit Curves ~')


def confusion_rates(cm):

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    N = fp + tn
    P = tp + fn

    tpr = tp / P
    fpr = fp / P
    fnr = fn / N
    tnr = tn / N

    rates = np.array([[tpr, fpr], [fnr, tnr]])

    return rates


def profit_curve(classifiers):
    for clf_class in classifiers:
        name, clf_class = clf_class[0], clf_class[1]
        clf = clf_class
        fit = clf.fit(X[train_index], y[train_index])
        probabilities = np.array([prob[0]
                                  for prob in fit.predict_proba(X[test_index])])
        profit = []

        indicies = np.argsort(probabilities)[::1]

        for idx in range(len(indicies)):
            pred_true = indicies[:idx]
            ctr = np.arange(indicies.shape[0])
            masked_prediction = np.in1d(ctr, pred_true)
            cm = confusion_matrix(y_test.astype(
                int), masked_prediction.astype(int))
            rates = confusion_rates(cm)

            profit.append(np.sum(np.multiply(rates, cb)))

        plt.plot((np.arange(len(indicies)) / len(indicies) * 100),
                 profit, label=name)
    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.show()


# %%
y_test = y[test_index].astype(float)

# Cost-Benefit Matrix
cb = np.array([[4, -5],
               [0, 0]])

# Define classifiers for comparison
classifiers = [("Logistic Regression", LR()),
               ("Gradient Boosting Classifier", GBC()),
               ("Support Vector Machines", SVC(probability=True)),
               ("Random Forest", RF(n_estimators=18)),
               ("K-nearest-neighbors:", KNN())
               ]

# Plot profit curves
profit_curve(classifiers)


# %%
forest = RF(n_estimators=18, random_state=4)
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

print(confusion_matrix(y[test_index], predictions))
print(accuracy_score(y[test_index], predictions))
print(classification_report(y[test_index], predictions))


# %%
# Grid Search and Hyper Parameters
rfc = RF(n_jobs=-1, max_features='sqrt',
         n_estimators=50, oob_score=True, random_state=4)

param_grid = {
    'n_estimators': [5, 10, 20, 40, 80, 160, 200],
    # 'min_samples_leaf': [1, 5, 10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X[train_index], y[train_index])

means = CV_rfc.cv_results_['mean_test_score']
stds = CV_rfc.cv_results_['std_test_score']

print(("Best: %f using %s with %s" %
       (CV_rfc.best_score_, CV_rfc.best_params_, CV_rfc.best_estimator_)))
for params, mean_score, scores in zip(CV_rfc.cv_results_['params'], means, stds):
    print(("%f (%f) with: %r" % (scores.mean(), scores.std(), params)))


# %%
forest = CV_rfc.best_estimator_
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

print(confusion_matrix(y[test_index], predictions))
print(accuracy_score(y[test_index], predictions))
print(classification_report(y[test_index], predictions))

print("Random forests senstivity analysis Train Data:")
plot_roc(X[train_index], y[train_index], CV_rfc.best_estimator_)
print("Random forests senstivity analysis Test Data:")
plot_roc(X[test_index], y[test_index], CV_rfc.best_estimator_)


# %%
print("~ Threshold Value ~")
clf = CV_rfc.best_estimator_

n_trials = 50
test_size_percent = 0.1

signals = X
labels = y

plot_data = []

train_signals, test_signals, train_labels, test_labels = train_test_split(
    signals, labels, test_size=test_size_percent)
clf.fit(train_signals, train_labels)
predictions = clf.predict_proba(test_signals)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    test_labels, predictions)
thresholds = np.append(thresholds, 1)

queue_rate = []
for threshold in thresholds:
    queue_rate.append((predictions >= threshold).mean())

plt.plot(thresholds, precision, color=sns.color_palette()[0])
plt.plot(thresholds, recall, color=sns.color_palette()[1])
plt.plot(thresholds, queue_rate, color=sns.color_palette()[2])

leg = plt.legend(('precision', 'recall', 'queue_rate'), frameon=True)
leg.get_frame().set_edgecolor('k')
plt.xlabel('threshold')
plt.ylabel('%')


# %%
clf = CV_rfc.best_estimator_

n_trials = 50
test_size_percent = 0.1

signals = X
labels = y

plot_data = []

for trial in range(n_trials):
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signals, labels, test_size=test_size_percent)
    clf.fit(train_signals, train_labels)
    predictions = clf.predict_proba(test_signals)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        test_labels, predictions)
    thresholds = np.append(thresholds, 1)

    queue_rate = []
    for threshold in thresholds:
        queue_rate.append((predictions >= threshold).mean())

    plot_data.append({
        'thresholds': thresholds,   'precision': precision,   'recall': recall,   'queue_rate': queue_rate
    })
# %%
for p in plot_data:
    plt.plot(p['thresholds'], p['precision'],
             color=sns.color_palette()[0], alpha=0.5)
    plt.plot(p['thresholds'], p['recall'],
             color=sns.color_palette()[1], alpha=0.5)
    plt.plot(p['thresholds'], p['queue_rate'],
             color=sns.color_palette()[2], alpha=0.5)

leg = plt.legend(('precision', 'recall', 'queue_rate'), frameon=True)
leg.get_frame().set_edgecolor('k')
plt.xlabel('threshold')


# %%
uniform_thresholds = np.linspace(0, 1, 101)

uniform_precision_plots = []
uniform_recall_plots = []
uniform_queue_rate_plots = []

for p in plot_data:
    uniform_precision = []
    uniform_recall = []
    uniform_queue_rate = []
    for ut in uniform_thresholds:
        index = bisect.bisect_left(p['thresholds'], ut)
        uniform_precision.append(p['precision'][index])
        uniform_recall.append(p['recall'][index])
        uniform_queue_rate.append(p['queue_rate'][index])

    uniform_precision_plots.append(uniform_precision)
    uniform_recall_plots.append(uniform_recall)
    uniform_queue_rate_plots.append(uniform_queue_rate)
# %%
quantiles = [0.1, 0.5, 0.9]
lower_precision, median_precision, upper_precision = mstats.mquantiles(
    uniform_precision_plots, quantiles, axis=0)
lower_recall, median_recall, upper_recall = mstats.mquantiles(
    uniform_recall_plots, quantiles, axis=0)
lower_queue_rate, median_queue_rate, upper_queue_rate = mstats.mquantiles(
    uniform_queue_rate_plots, quantiles, axis=0)

plt.plot(uniform_thresholds, median_precision)
plt.plot(uniform_thresholds, median_recall)
plt.plot(uniform_thresholds, median_queue_rate)

plt.fill_between(uniform_thresholds, upper_precision, lower_precision,
                 alpha=0.5, linewidth=0, color=sns.color_palette()[0])
plt.fill_between(uniform_thresholds, upper_recall, lower_recall,
                 alpha=0.5, linewidth=0, color=sns.color_palette()[1])
plt.fill_between(uniform_thresholds, upper_queue_rate, lower_queue_rate,
                 alpha=0.5, linewidth=0, color=sns.color_palette()[2])

leg = plt.legend(('precision', 'recall', 'queue_rate'), frameon=True)
leg.get_frame().set_edgecolor('k')
plt.xlabel('threshold')
plt.ylabel('%')


# %%
uniform_thresholds = np.linspace(0, 1, 101)

uniform_payout_plots = []

n = 10000
success_payoff = 4
case_cost = 5

for p in plot_data:
    uniform_payout = []
    for ut in uniform_thresholds:
        index = bisect.bisect_left(p['thresholds'], ut)
        precision = p['precision'][index]
        queue_rate = p['queue_rate'][index]

        payout = n*queue_rate*(precision*100 - case_cost)
        uniform_payout.append(payout)

    uniform_payout_plots.append(uniform_payout)

quantiles = [0.1, 0.5, 0.9]
lower_payout, median_payout, upper_payout = mstats.mquantiles(
    uniform_payout_plots, quantiles, axis=0)

plt.plot(uniform_thresholds, median_payout, color=sns.color_palette()[4])
plt.fill_between(uniform_thresholds, upper_payout, lower_payout,
                 alpha=0.5, linewidth=0, color=sns.color_palette()[4])

max_ap = uniform_thresholds[np.argmax(median_payout)]
plt.vlines([max_ap], -100000, 150000, linestyles='--')
plt.ylim(-100000, 150000)

leg = plt.legend(
    ('payout ($)', 'median argmax = {:.2f}'.format(max_ap)), frameon=True)
leg.get_frame().set_edgecolor('k')
plt.xlabel('threshold')
plt.title("Payout as a Function of Threshold")
plt.ylabel('$')


# %%
print(np.max(median_payout))


# %%
# Scoring Model
def ChurnModel(df, clf):
    # Convert yes no columns to bool
    # yes_no_cols already known, stored as a global variable
    df[yes_no_cols] = df[yes_no_cols] == 'yes'
    # features already known, stored as a global variable
    X = df[features].to_numpy().astype(np.float)
    X = scaler.transform(X)

    """
    Calculates probability of churn and expected loss, 
    and gathers customer's contact info
    """
    # Collect customer meta data
    response = df[['Area Code', 'Phone']]
    charges = ['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge']
    response['customer_worth'] = df[charges].sum(axis=1)

    # Make prediction
    churn_prob = clf.predict_proba(X)
    response['churn_prob'] = churn_prob[:, 1]

    # Calculate expected loss
    response['expected_loss'] = response['churn_prob'] * \
        response['customer_worth']
    response = response.sort_values('expected_loss', ascending=False)

    # Return response DataFrame
    return response


# %%
# Simulated new data
df = pd.read_csv("data/churn.csv")
train_index, test_index = train_test_split(df.index, random_state=4)
test_df = df.iloc[test_index]

# Apply new data to the model
ChurnModel(test_df, CV_rfc.best_estimator_)
# %%
InteractiveShell.ast_node_interactivity = "all"