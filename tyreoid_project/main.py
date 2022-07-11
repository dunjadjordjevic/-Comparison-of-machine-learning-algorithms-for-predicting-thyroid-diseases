import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def clear_dataset(thyroidDataset):

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
        'referral_source'],
        axis=1, inplace=True)

    ############################### --> Missing data CHECK <-- #########################################################
    # From description of the dataset we can see that there are columns with null values
    print(thyroidDataset.describe())

    # Printing total number of null value entries per column and percantage
    total = thyroidDataset.isnull().sum().sort_values(ascending=False)
    percent = (thyroidDataset.isnull().sum() / thyroidDataset.isnull().count()).sort_values(ascending=False)
    missing_data_table = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
    print(missing_data_table.head(10))

    # TBG has 96% of null data -> remove this column from dataset because is too important for analyzing
    thyroidDataset.drop(['TBG'], axis=1, inplace=True)
    # Fill with median values all entries that have null values for following columns: TSH, T3, TT4, T4U, FTI
    thyroidDataset['TSH'].fillna(thyroidDataset['TSH'].median(), inplace=True)
    thyroidDataset['T3'].fillna(thyroidDataset['T3'].median(), inplace=True)
    thyroidDataset['TT4'].fillna(thyroidDataset['TT4'].median(), inplace=True)
    thyroidDataset['T4U'].fillna(thyroidDataset['T4U'].median(), inplace=True)
    thyroidDataset['FTI'].fillna(thyroidDataset['FTI'].median(), inplace=True)

    # Checking how many entries have null value for sex, and pregnant column set to true
    # There is 4 rows with 'sex' null value and 'pregnant' true value
    # Setting those 4 entries to have 'sex' value 0 (female)
    print(thyroidDataset.loc[thyroidDataset['sex'].isnull() & thyroidDataset['pregnant'] == 1])
    thyroidDataset.loc[thyroidDataset['sex'].isnull() & thyroidDataset['pregnant'] == 1, 'sex'] = 0

    print('Thyroid dataset null values by column after updating')
    print(thyroidDataset.isnull().sum())


    ######################################## --> MAX/MIN unexpected values check <-- ##################################
    # Checking max/min values for age column
    print('Max value for age column')
    print(thyroidDataset['age'].max())
    print('Min value for age column')
    print(thyroidDataset['age'].min())

    # Updating entries with max and min values greater/lower than expected
    print(thyroidDataset[thyroidDataset.age > 100])
    thyroidDataset['age'] = np.where((thyroidDataset.age > 100), 100, thyroidDataset.age)

    ##################################### --> Mapping boolean values <-- ###############################################
    # --> Updating dataset to map true->1 and false->0  <--
    thyroidDataset.replace({'f': 0, 't': 1}, inplace=True)
    thyroidDataset['sex'].replace({'F': 0, 'M': 1}, inplace=True)

    ###################################### --> Removing duplicates <-- ################################################
    print(thyroidDataset.duplicated(keep='first').sum())
    print(thyroidDataset.size)
    thyroidDataset.drop_duplicates()
    print(thyroidDataset.size)
    # No duplicates found

    missingness = thyroidDataset.isnull().sum().sum() / thyroidDataset.count().sum()
    print('Overall missingness of thyroidDF after cleaning is: {:.2f}%'.format(missingness * 100))

    ################################## Checking unique values for target page #########################################
    print("Unique values for columns: ")
    print(thyroidDataset['target'].unique())

    print('Thyroid dataset cleaning END')

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
    thyroidDataset = thyroidDataset.replace({'negative': 0, 'hypothyroid': 1, 'hyperthyroid': 2})

    if thyroidDataset['target'].isnull().sum():
        thyroidDataset.dropna(subset=['target'], inplace=True)
        print('Null values from target column have been removed\n')
    else:
        print('There are no null values in target column\n')

    print(thyroidDataset['target'].unique())

    print('Remapping has been done on thyroid dataset!')

def add_more_data(thyroidDataset):

    print('Before implementing SMOTE: ')
    print(thyroidDataset['target'].value_counts())

    # SMOTE - Synthetic Minority Over-sampling Technique - adding more entries
    y = thyroidDataset['target']
    X = thyroidDataset.drop(columns=['target'], axis=1)
    oversample = SMOTE()
    X_smoted, y_smoted = oversample.fit_resample(X, y)
    y_smoted = pd.Series(y_smoted)
    y_smoted.value_counts()
    thyroidDataset = pd.concat([X_smoted, y_smoted], axis='columns')
    print('After implementing SMOTE: ')
    print(thyroidDataset['target'].value_counts())


#######################################################################################################################

# Reading thyroid dataset from CSV
thyroidDataset = pd.read_csv('resources/thyroidDF.csv')
# print('Thyroid dataset information')
thyroidDataset.info()
columns = thyroidDataset.columns
# print(columns)

clear_dataset(thyroidDataset)
remap_target_data(thyroidDataset)

# Some classes are more represented than others, so we need to add more entries in existing dataset
add_more_data(thyroidDataset)









