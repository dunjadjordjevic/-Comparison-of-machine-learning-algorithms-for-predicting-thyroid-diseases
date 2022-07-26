import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from tyreoid_project.models.algorithms.decision_tree import calculate_gini
from tyreoid_project.models.algorithms.decision_tree import calculate_entropy
from tyreoid_project.models.evaluator import Evaluator
from sklearn.metrics import confusion_matrix
from tyreoid_project.models.post_pruning import *

from models import evaluator
from tyreoid_project.models.plot_functions import create_plot


def clear_dataset(thyroidDataset, columns):

    print('Thyroid dataset cleaning START')

    missingness = thyroidDataset.isnull().sum().sum() / thyroidDataset.count().sum()
    print('Overall missingness of thyroidDF before cleaning is: {:.2f}%'.format(missingness * 100))

    # --> Removing unnecessary columns <--
    # Some columns are not necessary, so we can remove them
    thyroidDataset.drop([
        'TSH_measured',
        'T3_measured',
        'TT4_measured',
        'T4U_measured',
        'FTI_measured',
        'TBG_measured',
        'referral_source',
        'patient_id'],
        axis=1, inplace=True)

    columns = columns.drop([
        'TSH_measured',
        'T3_measured',
        'TT4_measured',
        'T4U_measured',
        'FTI_measured',
        'TBG_measured',
        'referral_source',
        'patient_id'])

    ############################### --> Missing data CHECK <-- #########################################################
    # From description of the dataset we can see that there are columns with null values
    print(thyroidDataset.describe())

    # Printing total number of null value entries per column and percantage
    total = thyroidDataset.isnull().sum().sort_values(ascending=False)
    percent = (thyroidDataset.isnull().sum() / thyroidDataset.isnull().count()).sort_values(ascending=False)
    missing_data_table = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
    #print(missing_data_table.head(10))

    # TBG has 96% of null data -> remove this column from dataset because is too important for analyzing
    #For every column that has missingness more than 80%, remove it from DF
    for idx, entry in missing_data_table.iterrows():
        if ((entry['Percentage']*100) >= 80):
            thyroidDataset.drop([entry.name], axis=1, inplace=True)
            columns = columns.drop([entry.name])

    # Fill with median values all entries that have null values for following columns: TSH, T3, TT4, T4U, FTI
    thyroidDataset['TSH'].fillna(thyroidDataset['TSH'].median(), inplace=True)
    thyroidDataset['T3'].fillna(thyroidDataset['T3'].median(), inplace=True)
    thyroidDataset['TT4'].fillna(thyroidDataset['TT4'].median(), inplace=True)
    thyroidDataset['T4U'].fillna(thyroidDataset['T4U'].median(), inplace=True)
    thyroidDataset['FTI'].fillna(thyroidDataset['FTI'].median(), inplace=True)

    ##################################### --> Mapping boolean values <-- ###############################################

    # --> Updating dataset to map true->1 and false->0  <--
    thyroidDataset.replace({'f': 0, 't': 1}, inplace=True)
    thyroidDataset['sex'].replace({'F': 0, 'M': 1}, inplace=True)

    # Checking how many entries have null value for sex, and pregnant column set to true
    # There is 4 rows with 'sex' null value and 'pregnant' true value in thyroid dataset
    # Setting those 4 entries to have 'sex' value 0 (female)
    # print(thyroidDataset.loc[thyroidDataset['sex'].isnull() & thyroidDataset['pregnant'] == 1])
    if (thyroidDataset.loc[thyroidDataset['sex'].isnull() & thyroidDataset['pregnant'] == 1].shape[0] > 0):
        thyroidDataset.loc[thyroidDataset['sex'].isnull() & thyroidDataset['pregnant'] == 1, 'sex'] = 0

    # Remove entries where 'sex' is nan value
    thyroidDataset.dropna(subset=['sex'], inplace=True)

    #print('Thyroid dataset null values by column after updating')
    #print(thyroidDataset.isnull().sum())

    ######################################## --> MAX/MIN unexpected values check <-- ##################################

    # Checking max/min values for age column
    print('Max value for age column')
    print(thyroidDataset['age'].max())
    print('Min value for age column')
    print(thyroidDataset['age'].min())

    # Updating entries with max and min values greater/lower than expected
    if (thyroidDataset[thyroidDataset.age > 100].shape[0] > 0):
        thyroidDataset['age'] = np.where((thyroidDataset.age > 100), 100, thyroidDataset.age)

    ###################################### --> Removing duplicates <-- ################################################

    if (thyroidDataset.duplicated(keep='first').sum() > 0):
        thyroidDataset = thyroidDataset.drop_duplicates()

    missingness = thyroidDataset.isnull().sum().sum() / thyroidDataset.count().sum()
    #print('Overall missingness of thyroidDF after cleaning is: {:.2f}%'.format(missingness * 100))

    ################################## Checking unique values for target page #########################################

    #print("Unique values for columns: ")
    #print(thyroidDataset['target'].unique())

    print('Thyroid dataset cleaning END')

    return thyroidDataset, columns

def remap_target_data(thyroidDataset):

    # Target column in thyroid dataset include Strings which represents exact diagnose with multiple
    # parameters for each patient; This column will be mapped to have only 3 possible values (output classes):
    # hyperthyroid, hypothyroid, negative
    # so classification process will be easier done
    # Mapping for values will be convert to int values: negative -> 0, hypo -> 1, hyper -> 2

    diagnoses = {'-': 'negative',
                 'A': 'hyperthyroid',
                 'B': 'hyperthyroid',
                 'C': 'hyperthyroid',
                 'D': 'hyperthyroid',
                 'E': 'hypothyroid',
                 'F': 'hypothyroid',
                 'G': 'hypothyroid',
                 'H': 'hypothyroid'}

    thyroidDataset['target'] = thyroidDataset['target'].map(diagnoses)
    thyroidDataset.dropna(subset=['target'], inplace=True) #remove all data that are not in those 3 classes
    thyroidDataset = thyroidDataset.replace({'negative': 0, 'hypothyroid': 1, 'hyperthyroid': 2})
    thyroidDataset['target'] = thyroidDataset['target'].astype('int')

    if thyroidDataset['target'].isnull().sum():
        thyroidDataset.dropna(subset=['target'], inplace=True)
        print('Null values from target column have been removed\n')
    else:
        print('There are no null values in target column\n')

    #print(thyroidDataset['target'].unique())

    print('Remapping has been done on thyroid dataset!')

    return thyroidDataset

def add_more_data(thyroidDataset):

    print('Before implementing SMOTE: ')
    #print(thyroidDataset['target'].value_counts())

    # SMOTE - Synthetic Minority Over-sampling Technique - adding more entries
    y = thyroidDataset['target'].astype(str)
    X = thyroidDataset.drop(columns=['target'], axis=1)
    oversample = SMOTE()
    X_smoted, y_smoted = oversample.fit_resample(X, y)
    y_smoted = pd.Series(y_smoted)
    y_smoted.value_counts()
    thyroidDataset = pd.concat([X_smoted, y_smoted], axis='columns')
    y = thyroidDataset['target'].astype(int)
    print('After implementing SMOTE: ')
    print(thyroidDataset['target'].value_counts())
    return thyroidDataset

#######################################################################################################################

# Reading thyroid dataset from CSV
thyroidDataset = pd.read_csv('resources/thyroidDF.csv')

columns = thyroidDataset.columns

thyroidDataset, columns = clear_dataset(thyroidDataset, columns)
thyroidDataset = remap_target_data(thyroidDataset)

# Some classes are more represented than others, so we need to add more entries in existing dataset
thyroidDataset = add_more_data(thyroidDataset)

'''
# Correlation matrix for get information about weak, positive, negative correlations in DS for features

plt.figure(figsize=(12, 10))
cor = thyroidDataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.title('Correlation matrix after cleanining data')
plt.show()
'''

y = thyroidDataset['target']
x = thyroidDataset.drop('target', axis=1)

#Split data in two groups: 20% test, 80% train with shuffle method
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

number_of_records_in_training_set = len(x_train)
number_of_records_in_test_set = len(x_test)

'''

# Checking importance of each feature in DS

reg = LassoCV()
reg.fit(x_train, y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x_test,y_test))
coef = pd.Series(reg.coef_, index=x_train.columns)
print('Coefficients: ')
print(coef)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()

'''

######################################################  EVALUATION  ####################################################

evaluator = evaluator.Evaluator()
evaluator.evaluate(pd.concat([x_train, y_train], axis='columns'), columns)

######################################################  PREDICTION  ####################################################

# Call method for every entry in test set, and whole training set
# with specific parameters (which have been determined before as the best option)
# to get prediction, and then see what is metric:
# 1. Accuracy - number of correct predictions test set / Total number of elements in test set
# 2. Generate confusion matrix

NUMBER_OF_NEIGHBOURS_PREDICTED = 110
MIN_SAMPLE_PREDICTED = 2
MAX_DEPTH_PREDICTED = 5
CALCULATE_METRIC = calculate_entropy
NUMBER_OF_TREES_PREDICTED = 50
NUMBER_OF_DATA_IN_BOOTSTRAP_DATASET = 400
NUMBER_OF_FEATURES = 800

'''
tmp = pd.concat([x_train, y_train], axis='columns')
y_train_tmp = tmp['target'].astype(int)
w = Evaluator.regularization_coeffients(self=evaluator, X=x_train, target=y_train_tmp.values)


#Predict with KNN algorithm
knn_prediction_values = evaluator.predict_algorithm(pd.concat([x_train, y_train], axis='columns'),
                            pd.concat([x_test, y_test], axis='columns'),
                            columns,
                            'k_nearest_neighbours',
                            NUMBER_OF_NEIGHBOURS_PREDICTED,
                            w)

knn_prediction_values = [int(x) for x in knn_prediction_values]

#Generate confusion matrix for KNN
knn_cf_matrix = confusion_matrix(y_test.values.astype(int), knn_prediction_values)
knn_ax = sns.heatmap(knn_cf_matrix/np.sum(knn_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
knn_ax.set_title('KNN Confusion Matrix\n\n')
knn_ax.set_xlabel('\nPredicted Values')
knn_ax.set_ylabel('Actual Values ')
knn_ax.xaxis.set_ticklabels(['0', '1', '2'])
knn_ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()

'''

'''
#Predict with Decision tree algorithm
dt_prediction_values = evaluator.predict_algorithm(pd.concat([x_train, y_train], axis='columns'),
                            pd.concat([x_test, y_test], axis='columns'),
                            columns,
                            'decision_tree_algorithm',
                            MIN_SAMPLE_PREDICTED,
                            MAX_DEPTH_PREDICTED,
                            CALCULATE_METRIC)

dt_prediction_values = [int(x) for x in dt_prediction_values]

#Generate confusion matrix for DT
dt_cf_matrix = confusion_matrix(y_test.values.astype(int), dt_prediction_values)
dt_ax = sns.heatmap(dt_cf_matrix/np.sum(dt_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
dt_ax.set_title('DT Confusion Matrix\n\n')
dt_ax.set_xlabel('\nPredicted Values')
dt_ax.set_ylabel('Actual Values ')
dt_ax.xaxis.set_ticklabels(['0', '1', '2'])
dt_ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()

'''

'''
MAX_DEPTH_PREDICTED = 12
#Predict with Random forest algorithm
rf_prediction_values = evaluator.predict_algorithm(pd.concat([x_train, y_train], axis='columns'),
                            pd.concat([x_test, y_test], axis='columns'),
                            columns,
                            'random_forest_algorithm',
                            NUMBER_OF_TREES_PREDICTED,
                            NUMBER_OF_DATA_IN_BOOTSTRAP_DATASET,
                            NUMBER_OF_FEATURES,
                            MAX_DEPTH_PREDICTED,
                            CALCULATE_METRIC)

rf_prediction_values = [int(x) for x in rf_prediction_values]

#Generate confusion matrix for RT
rf_cf_matrix = confusion_matrix(y_test.values.astype(int), rf_prediction_values)
rf_ax = sns.heatmap(rf_cf_matrix/np.sum(rf_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
rf_ax.set_title('RF Confusion Matrix\n\n')
rf_ax.set_xlabel('\nPredicted Values')
rf_ax.set_ylabel('Actual Values ')
rf_ax.xaxis.set_ticklabels(['0', '1', '2'])
rf_ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()

'''

'''
#Predict with Naive Bayes classifier algorithm
nbc_prediction_values = evaluator.predict_algorithm(pd.concat([x_train, y_train], axis='columns'),
                            pd.concat([x_test, y_test], axis='columns'),
                            columns,
                            'naive_bayes_classifier_algorithm')

nbc_prediction_values = [int(x) for x in nbc_prediction_values]

#Generate confusion matrix for Naive Bayes classifier
nbc_cf_matrix = confusion_matrix(y_test.values.astype(int), nbc_prediction_values)
nbc_ax = sns.heatmap(nbc_cf_matrix/np.sum(nbc_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
nbc_ax.set_title('Naive Bayes classifier Ccnfusion Matrix\n\n')
nbc_ax.set_xlabel('\nPredicted Values')
nbc_ax.set_ylabel('Actual Values ')
nbc_ax.xaxis.set_ticklabels(['0', '1', '2'])
nbc_ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()
'''











