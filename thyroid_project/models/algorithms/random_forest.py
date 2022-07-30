import random
import numpy as np
import pandas as pd

from thyroid_project.models.algorithms.decision_tree import decision_tree_algorithm, classify_entry

'''
    Create predictions for all entries in dataset
'''
def decision_tree_predictions(test_df, tree):

    prediction_values = []

    for entry in test_df.iterrows():
        prediction_value = classify_entry(entry, tree)
        prediction_values.append(prediction_value)

    return prediction_values

def calculate_accuracy(predictions, targets):

    correct_predictions = predictions == targets
    return correct_predictions.mean()

def bootstrapping(train_df, number_of_data_in_bootstrap_dataset):

    bootstrap_entries_indexes = np.random.randint(0, len(train_df), number_of_data_in_bootstrap_dataset)
    boostrapped_dataset = train_df.iloc[bootstrap_entries_indexes]
    return boostrapped_dataset

def random_forest_algorithm(train_df, columns, number_of_trees, number_of_data_in_bootstrap_dataset, number_of_features, max_depth, metric_function):

    # forest is array of trees
    forest = []
    train_df = pd.DataFrame(data=train_df, columns=list(columns))

    for i in range(number_of_trees):
        df_bootstrapped = bootstrapping(train_df, number_of_data_in_bootstrap_dataset)
        tree = decision_tree_algorithm(df_bootstrapped, columns, max_depth=max_depth, random_subspace=number_of_features, metric_function=metric_function)
        forest.append(tree)

    return forest

def random_forest_predictions(test_df, forest):

    # dict where we are going to put for evry tree his predictions
    df_predictions = {}

    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        tree = forest[i]
        predictions_for_tree = decision_tree_predictions(test_df, tree)
        df_predictions[column_name] = predictions_for_tree

    df_predictions = pd.DataFrame(df_predictions)

    return df_predictions.mode(axis=1)[0]

def random_forest(train_set, test_set, columns, number_of_trees, number_of_data_in_bootstrap_dataset, number_of_features, max_depth, metric_function):

    forest = random_forest_algorithm(train_set, columns, number_of_trees, number_of_data_in_bootstrap_dataset, number_of_features, max_depth, metric_function)
    test_set = pd.DataFrame(data=test_set, columns=columns)
    predictions = random_forest_predictions(test_set, forest)
    #accuracy = calculate_accuracy(predictions, test_set.target)

    return predictions