import numpy as np
import pandas as pd
import random

def check_purity(dataset):

    target_column = dataset[:, -1]
    unique_classes = np.unique(target_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(dataset):

    target_column = dataset[:, -1]
    unique_classes, counts_unique_classes = np.unique(target_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(dataset, random_subspace):

    # dictionary where key is index of column, and value is list of potential splits
    potential_splits = {}
    _, number_of_columns = dataset.shape
    column_indexes = list(range(number_of_columns - 1))

    # for random forest tree algorithm
    if random_subspace and random_subspace <= len(column_indexes):
        column_indexes = random.sample(column_indexes, random_subspace)

    for column_index in column_indexes:
        values = dataset[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values

    return potential_splits


'''
    dataset - dataset for which we are doing the split
    split_column - column in dataset for which we are doing the split
    split_value  - value in values of mentioned column on which we will do the split
    return value - are two lists of rows - left where values are lower or eq to split_value and 
                right where values are greater than split_value
'''


def split_data(dataset, split_column, split_value):

    split_column_values = dataset[:, split_column]
    data_left = dataset[split_column_values <= split_value]
    data_right = dataset[split_column_values > split_value]

    return data_left, data_right


def calculate_overall_metric(data_below, data_above, metric_function):

    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric = (p_data_below * metric_function(data_below)
                      + p_data_above * metric_function(data_above))

    return overall_metric


def calculate_entropy(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

def calculate_gini(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities*probabilities)

    return gini


def determine_best_split(dataset, potential_splits, metric_function):

    first_iteration = True

    # loop through all keys in dictionary potential_splits
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_left, data_right = split_data(dataset, split_column=column_index, split_value=value)
            current_overall_metric = calculate_overall_metric(data_left, data_right, metric_function=metric_function)
            if (first_iteration == True) or (current_overall_metric <= best_overall_metric):

                first_iteration = False
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


'''
    df - dataframe from which we would like to create a tree
    counter - number of calls of function decision_tree_algorithm
    min_sample - finishing point, representing minimal number of elements in group for node which can 
    not be splited anymore (the greater the min_sample is, the lower number of layers tree will have :))
    max_depth - number of depth for tree, representing number of questions in dictionary
    random_subspace - number of features which we want to use in random forest algorithm
                      (default value is None for regular classification decision tree algorithm)
'''


def decision_tree_algorithm(df_copy, columns, counter=0, min_samples=2, max_depth=5, random_subspace=None, metric_function=calculate_entropy):

    # first call of function is when counter = 0
    # dataset je numpy to array
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = columns
        df = pd.DataFrame(df_copy, columns=columns)
        dataset = df.values
    else:
        dataset = df_copy

    if (check_purity(dataset) or (len(dataset) < min_samples) or counter == max_depth):
        # base part
        classification = classify_data(dataset)
        return classification
    else:
        # recursive_part
        counter += 1
        potential_splits = get_potential_splits(dataset, random_subspace)
        split_column, split_value = determine_best_split(dataset, potential_splits, metric_function)
        data_left, data_right = split_data(dataset, split_column, split_value)

        if len(data_left) == 0 or len(data_right) == 0:
            classification = classify_data(dataset)
            return classification

        # instantiate sub-tree
        # create key as question
        attribute_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(attribute_name, split_value)
        sub_tree = {question: []}

        # create value as list of answers
        yes_answer = decision_tree_algorithm(data_left, columns, counter, min_samples, max_depth, random_subspace, metric_function=metric_function)
        no_answer = decision_tree_algorithm(data_right, columns, counter, min_samples, max_depth, random_subspace, metric_function=metric_function)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


'''
    entry - it's one entry from test dataset
'''
def classify_entry(entry, tree):

    if not isinstance(tree, dict):
        return tree

    question = list(tree.keys())[0]
    attribute_name, comparasion_operator, value = question.split()

    # ask question
    if isinstance(entry, pd.Series):
        tmp = entry
    else:
        tmp = entry[1]

    if tmp[attribute_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # going through the tree and checking in which branch we should go down
    if not isinstance(answer, dict):
        return answer
    else:
        return classify_entry(entry, answer)

def classification_decision_tree(train_set, test_set, columns, min_samples=2, max_depth=5, metric_function=calculate_entropy):

    prediction_values = list()
    tree = decision_tree_algorithm(train_set, columns, min_samples=min_samples, max_depth=max_depth, metric_function=metric_function)
    print('Tree before post-pruning:\n')
    print(tree)
    print('Tree after post-pruning:\n')
    from tyreoid_project.models.post_pruning import post_pruning
    tree_pruned = post_pruning(tree, train_set, test_set, columns)
    print(tree_pruned)

    test_set = pd.DataFrame(data=test_set, columns=columns)
    for entry in test_set.iterrows():
        prediction_value = classify_entry(entry, tree_pruned)
        prediction_values.append(prediction_value)

    return prediction_values
