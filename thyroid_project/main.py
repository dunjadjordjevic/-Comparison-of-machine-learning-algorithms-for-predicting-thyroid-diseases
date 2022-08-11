import pandas as pd
import seaborn as sns
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

import matplotlib
import thyroid_project
from thyroid_project.models.evaluator import *
from sklearn.metrics import confusion_matrix
from thyroid_project.models.post_pruning import *

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
    print(thyroidDataset['target'].value_counts())

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


def generate_testing_random_datasets(thyroidDataset):

    listOfDatasets = []
    numberOfEntriesInInitialDataset = thyroidDataset.shape[0]
    print('Number of entries in dataset is: ', numberOfEntriesInInitialDataset)

    for i in range(1, 10, 1):
        newDataset = thyroidDataset.sample(n=randint(100, numberOfEntriesInInitialDataset), random_state=1)
        newDataset.to_csv('thyroid_project/resources/generated_datasets/test_dataset_' + str(i) + '.csv')
        listOfDatasets.append(newDataset)

    return listOfDatasets


#######################################################################################################################

# For the name of the CSV file, get CSV file, convert to DF, do cleaning, remapping, adding data and then
# do evaluation/prediction with specific algorithm depending on input parameters

def analyze_dataset(dataset, x_train, y_train, x_test, y_test):

    # Correlation matrix for get information about weak, positive, negative correlations in DS for features
    plt.figure(figsize=(18, 16))
    cor = dataset.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.title("Correlation matrix for thyroid dataset")
    plt.savefig('thyroid_project/resources/generated_images/thyroid_ds_attribute_correlation_matrix.png')
    #plt.show()
    plt.clf()
    plt.cla()

    # Checking importance of each feature in DS
    reg = LassoCV()
    reg.fit(x_train, y_train)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(x_test, y_test))
    coef = pd.Series(reg.coef_, index=x_train.columns)
    print('Coefficients: ')
    print(coef)
    imp_coef = coef.sort_values()
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.savefig('thyroid_project/resources/generated_images/thyroid_ds_attribute_importance.png')
    #plt.show()
    plt.clf()
    plt.cla()


def upload_CSV_file(nameOfFile):

    if (nameOfFile == 'thyroidDF'):
        dataset = pd.read_csv('thyroid_project/resources/' + nameOfFile + '.csv')
    else:
        dataset = pd.read_csv('thyroid_project/resources/generated_datasets/' + nameOfFile + '.csv')

    columns = dataset.columns
    listOfDatasets = list()

    if (nameOfFile == 'thyroidDF'):

        #Evaluation/Prediction for starting dataset
        dataset, columns = clear_dataset(dataset, columns)
        dataset = remap_target_data(dataset)

        # Some classes are more represented than others, so we need to add more entries in existing dataset
        dataset = add_more_data(dataset)

    return dataset, columns, listOfDatasets


def remove_dominant_attributes(train_dataset, columns):

    helper = train_dataset
    modes = helper.mode()

    print('Number of columns BEFORE deletition of dominant attributes is:', helper.shape[1])
    dominant_attributes = list()
    for index in range(0, len(columns) - 1):
        if (helper[columns[index]].value_counts().values[0] >= (0.95 * helper.shape[0])):
            print('Dominant attribute has been found - ', columns[index])
            dominant_attributes.append(columns[index])

    print('Dominant attributes will be removed from dataset')

    return dominant_attributes

def predictDataWithMetrics(nameOfFileForDataset, nameOfAlgorithm, columnsForDataset=None, x_train=None, y_train=None, x_test=None, y_test=None):

    evaluator1 = thyroid_project.models.evaluator.Evaluator()
    datasetWithPrediction = None
    accuracy = None
    accuracy_metric, precision, recall, f1score = None, None, None, None

    dataset, columns, listOfDatasets = upload_CSV_file(nameOfFileForDataset)
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    if 'Unnamed: 0' in columns:
        columns = columns.drop(['Unnamed: 0'])

    #in case thyroid_dataset
    if (columnsForDataset is not None):
        columns = columnsForDataset

    #in case not thyroid_dataset
    if (x_test is None and y_test is None):
        y_test = dataset['target']
        x_test = dataset.drop('target', axis=1)


    # Split data in two groups: 20% test, 80% train with shuffle method
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    if (nameOfFileForDataset == 'thyroidDF'):
        analyze_dataset(dataset, x_train, y_train, x_test, y_test)

    number_of_records_in_training_set = len(x_train)
    number_of_records_in_test_set = len(x_test)

    # Putting parameter values for ML algorithms set by training process
    NUMBER_OF_NEIGHBOURS_PREDICTED = 109
    MIN_SAMPLE_PREDICTED = 2
    MAX_DEPTH_PREDICTED = 5
    CALCULATE_METRIC = calculate_entropy
    NUMBER_OF_TREES_PREDICTED = 50
    NUMBER_OF_DATA_IN_BOOTSTRAP_DATASET = 400
    NUMBER_OF_FEATURES = 800

    print('Prediction for algorithm: ', nameOfAlgorithm)
    if(nameOfAlgorithm == 'k_nearest_neighbours'):

        # Predict with KNN algorithm
        tmp = pd.concat([x_train, y_train], axis='columns')
        y_train_tmp = tmp['target'].astype(int)
        w = Evaluator.regularization_coeffients(self=evaluator1, X=x_train, target=y_train_tmp.values)

        predictions, accuracy_metric, precision, recall, f1score = evaluator1.predict_algorithm(
            pd.concat([x_train, y_train], axis='columns'),
            pd.concat([x_test, y_test], axis='columns'),
            columns,
            'k_nearest_neighbours',
            NUMBER_OF_NEIGHBOURS_PREDICTED,
            w)

        predictions = [int(x) for x in predictions]

        # Generate confusion matrix for KNN
        knn_cf_matrix = confusion_matrix(y_test.values.astype(int), predictions)
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 16})
        knn_ax = sns.heatmap(knn_cf_matrix / np.sum(knn_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        knn_ax.set_title('KNN Confusion Matrix\n\n')
        knn_ax.set_xlabel('\nPredicted Values')
        knn_ax.set_ylabel('Actual Values ')
        knn_ax.xaxis.set_ticklabels(['0', '1', '2'])
        knn_ax.yaxis.set_ticklabels(['0', '1', '2'])
        plt.savefig('thyroid_project/resources/generated_images/confusion_matrix.png')
        #plt.show()
        plt.clf()
        plt.cla()

    elif (nameOfAlgorithm == 'decision_tree_algorithm'):

        # Predict with Decision tree algorithm
        predictions, accuracy_metric, precision, recall, f1score = evaluator1.predict_algorithm(
            pd.concat([x_train, y_train], axis='columns'),
            pd.concat([x_test, y_test], axis='columns'),
            columns,
            'decision_tree_algorithm',
            MIN_SAMPLE_PREDICTED,
            MAX_DEPTH_PREDICTED,
            CALCULATE_METRIC)

        predictions = [int(x) for x in predictions]

        # Generate confusion matrix for DT
        dt_cf_matrix = confusion_matrix(y_test.values.astype(int), predictions)
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 16})
        dt_ax = sns.heatmap(dt_cf_matrix / np.sum(dt_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        dt_ax.set_title('DT Confusion Matrix\n\n')
        dt_ax.set_xlabel('\nPredicted Values')
        dt_ax.set_ylabel('Actual Values ')
        dt_ax.xaxis.set_ticklabels(['0', '1', '2'])
        dt_ax.yaxis.set_ticklabels(['0', '1', '2'])
        plt.savefig('thyroid_project/resources/generated_images/confusion_matrix.png')
        #plt.show()
        plt.clf()
        plt.cla()


    elif (nameOfAlgorithm == 'random_forest_algorithm'):

        MAX_DEPTH_PREDICTED = 12

        # Predict with Random forest algorithm
        predictions, accuracy_metric, precision, recall, f1score = evaluator1.predict_algorithm(
            pd.concat([x_train, y_train], axis='columns'),
            pd.concat([x_test, y_test], axis='columns'),
            columns,
            'random_forest_algorithm',
            NUMBER_OF_TREES_PREDICTED,
            NUMBER_OF_DATA_IN_BOOTSTRAP_DATASET,
            NUMBER_OF_FEATURES,
            MAX_DEPTH_PREDICTED,
            CALCULATE_METRIC)

        predictions = [int(x) for x in predictions]

        # Generate confusion matrix for RT
        rf_cf_matrix = confusion_matrix(y_test.values.astype(int), predictions)
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 16})
        rf_ax = sns.heatmap(rf_cf_matrix / np.sum(rf_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        rf_ax.set_title('RF Confusion Matrix\n\n')
        rf_ax.set_xlabel('\nPredicted Values')
        rf_ax.set_ylabel('Actual Values ')
        rf_ax.xaxis.set_ticklabels(['0', '1', '2'])
        rf_ax.yaxis.set_ticklabels(['0', '1', '2'])
        plt.savefig('thyroid_project/resources/generated_images/confusion_matrix.png')
        #plt.show()
        plt.clf()
        plt.cla()

    elif (nameOfAlgorithm == 'naive_bayes_classifier_algorithm'):

        # Predict with Naive Bayes classifier algorithm
        predictions, accuracy_metric, precision, recall, f1score = evaluator1.predict_algorithm(
            pd.concat([x_train, y_train], axis='columns'),
            pd.concat([x_test, y_test], axis='columns'),
            columns,
             'naive_bayes_classifier_algorithm')

        nbc_prediction_values = [int(x) for x in predictions]

        # Generate confusion matrix for Naive Bayes classifier
        nbc_cf_matrix = confusion_matrix(y_test.values.astype(int), predictions)
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 16})
        nbc_ax = sns.heatmap(nbc_cf_matrix / np.sum(nbc_cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        nbc_ax.set_title('Naive Bayes classifier Confusion Matrix\n\n')
        nbc_ax.set_xlabel('\nPredicted Values')
        nbc_ax.set_ylabel('Actual Values ')
        nbc_ax.xaxis.set_ticklabels(['0', '1', '2'])
        nbc_ax.yaxis.set_ticklabels(['0', '1', '2'])
        plt.savefig('thyroid_project/resources/generated_images/confusion_matrix.png')
        #plt.show()
        plt.clf()
        plt.cla()

    datasetWithPrediction = x_test
    datasetWithPrediction = datasetWithPrediction.assign(predicted_values=predictions)
    datasetWithPrediction = datasetWithPrediction.assign(actual_values=y_test)
    datasetWithPrediction.to_csv('thyroid_project/resources/generated_prediction_datasets/prediction_dataset.csv')

    return datasetWithPrediction, columns, accuracy_metric, precision, recall, f1score

########################################################################################################################
######################################################  EVALUATION  ####################################################

def prepareThyroidDataset():

    thyroidDataset, columns, listOfDatasets = upload_CSV_file('thyroidDF')

    '''
    # Correlation matrix for get information about weak, positive, negative correlations in DS for features
    plt.figure(figsize=(18, 16))
    cor = thyroidDataset.corr()
    print(cor.iloc[-1:]) #dataframe with last target row
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.title('Correlation matrix BEFORE cleanining dominant attributes')
    plt.savefig('thyroid_project/resources/generated_images/correlation_matrix_thyroidDF_before_cleaning.png')
    #plt.show()
    plt.clf()
    plt.cla()
    '''

    y = thyroidDataset['target']
    x = thyroidDataset.drop('target', axis=1)

    #Split data in two groups: 20% test, 80% train with shuffle method
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    return thyroidDataset, columns, x_train, x_test, y_train, y_test

def mainMethod(thyroidDataset, columns, x_train, x_test, y_train, y_test, evaluate):

    global evaluator
    evaluator2 = thyroid_project.models.evaluator.Evaluator()

    if (evaluate == True):

        clf = RandomForestClassifier(n_estimators = 100,
                                     bootstrap=True,
                                     max_depth=None,
                                     n_jobs=-1,
                                     min_samples_leaf=10,
                                     min_samples_split=2,
                                     random_state=123)
        clf.fit(x_train, y_train)
        coef = clf.feature_importances_
        ind = np.argsort(-coef)
        xtmp = range(x_train.shape[1])
        ytmp = coef[ind][:x_train.shape[1]]
        plt.title("Feature importances BEFORE cleaning")
        ax = plt.subplot()
        plt.barh(xtmp, ytmp, color='red')
        ax.set_yticks(xtmp)
        ax.set_yticklabels(columns[ind])
        plt.gca().invert_yaxis()
        plt.savefig('thyroid_project/resources/generated_images/feature_importances_thyroidDF_before_cleaning.png', bbox_inches="tight")
        plt.clf()
        plt.cla()

    remove_dominant_attributes_list = remove_dominant_attributes(pd.concat([x_train, y_train], axis='columns'), columns)
    thyroidDataset.drop(remove_dominant_attributes_list, axis=1, inplace=True)
    x_train.drop(remove_dominant_attributes_list, axis=1, inplace=True)
    x_test.drop(remove_dominant_attributes_list, axis=1, inplace=True)
    columns = columns.drop(remove_dominant_attributes_list)

    if (evaluate == True):

        thyroidDataset['target'] = [int(x) for x in thyroidDataset['target']]
        thyroidDataset['sex'] = [int(x) for x in thyroidDataset['sex']]

        thyroidDataset.hist(figsize=(20,20), xrot=-45)
        plt.savefig('thyroid_project/resources/generated_images/attributes_distribution_thyroidDF_after_cleaning.png')
        plt.clf()
        plt.cla()

        # Generate list of 10 testing CSV files, where every CSV file contains dataframe with 22 columns and random number
        # of rows from 100 to number of entries in thyroidDataset (19428) with randomly entries get from that dataset
        listOfDatasets = generate_testing_random_datasets(thyroidDataset)

        # Checking importance of each feature in DS
        reg = LassoCV()
        reg.fit(x_train, y_train)
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        # Second time check
        # Correlation matrix for get information about weak, positive, negative correlations in DS for features
        plt.figure(figsize=(18, 16))
        cor = thyroidDataset.corr()
        print(cor.iloc[-1:]) #dataframe with last target row
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.title('Correlation matrix AFTER cleanining dominant attributes')
        plt.savefig('thyroid_project/resources/generated_images/correlation_matrix_thyroidDF_after_cleaning.png')
        #plt.show()
        plt.clf()
        plt.cla()

        clf = RandomForestClassifier(n_estimators = 100,
                                     bootstrap=True,
                                     max_depth=None,
                                     n_jobs=-1,
                                     min_samples_leaf=10,
                                     min_samples_split=2,
                                     random_state=123)
        clf.fit(x_train, y_train)
        coef = clf.feature_importances_
        ind = np.argsort(-coef)
        xtmp = range(x_train.shape[1])
        ytmp = coef[ind][:x_train.shape[1]]
        plt.title("Feature importances AFTER cleaning")
        ax = plt.subplot()
        plt.barh(xtmp, ytmp, color='red')
        ax.set_yticks(xtmp)
        ax.set_yticklabels(columns[ind])
        plt.gca().invert_yaxis()
        plt.savefig('thyroid_project/resources/generated_images/feature_importances_thyroidDF_after_cleaning.png')
        plt.clf()
        plt.cla()

        evaluator2.evaluate(pd.concat([x_train, y_train], axis='columns'), columns)

    return thyroidDataset, columns, x_train, x_test, y_train, y_test

thyroidDataset, columns, x_train, x_test, y_train, y_test = prepareThyroidDataset()
thyroidDataset, columns, x_train, x_test, y_train, y_test = \
    mainMethod(thyroidDataset, columns, x_train, x_test, y_train, y_test, evaluate=False)

########################################################################################################################
######################################################  PREDICTION  ####################################################

datasetWithPrediction, columns, accuracy_metric, precision, recall, f1score = \
     predictDataWithMetrics('thyroidDF', 'k_nearest_neighbours', columns, x_train, y_train, x_test, y_test)

########################################################################################################################
########################################################################################################################