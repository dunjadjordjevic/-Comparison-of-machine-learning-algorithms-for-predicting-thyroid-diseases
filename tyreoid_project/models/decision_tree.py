import numpy as np
import pandas as pd

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


def get_potential_splits(dataset):

    # dictionary where key is index of column, and value is list of potential splits
    potential_splits = {}

    _, number_of_columns = dataset.shape
    for column_index in range(number_of_columns - 1):
        # for each column we should create one entry in dictionary potential_splits
        potential_splits[column_index] = []
        # for each column we are storing values for that column in values variable
        values = dataset[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                potential_splits[column_index].append(potential_split)

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


def determine_best_split(dataset, potential_splits):

    first_iteration = True

    # loop through all keys in dictionary potential_splits
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_left, data_right = split_data(dataset, split_column=column_index, split_value=value)
            current_overall_metric = calculate_overall_metric(data_left, data_right, metric_function=calculate_entropy)
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
'''


def decision_tree_algorithm(df_copy, columns, counter=0, min_samples=2, max_depth=5):


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
        potential_splits = get_potential_splits(dataset)
        split_column, split_value = determine_best_split(dataset, potential_splits)
        data_left, data_right = split_data(dataset, split_column, split_value)

        # instantiate sub-tree
        # create key as question
        attribute_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(attribute_name, split_value)
        sub_tree = {question: []}

        # create value as list of answers
        yes_answer = decision_tree_algorithm(data_left, columns, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_right, columns, counter, min_samples, max_depth)

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

    question = list(tree.keys())[0]
    attribute_name, comparasion_operator, value = question.split()

    # ask question
    if entry[1][attribute_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # going through the tree and checking in which branch we should go down
    if not isinstance(answer, dict):
        return answer
    else:
        return classify_entry(entry, answer)


def classification_decision_tree(train_set, test_set, columns, min_samples=2, max_depth=5):

    prediction_values = list()
    tree = decision_tree_algorithm(train_set, columns, min_samples=min_samples, max_depth=max_depth)
    print(tree)

    test_set = pd.DataFrame(data=test_set, columns=columns)
    for entry in test_set.iterrows():
        prediction_value = classify_entry(entry, tree)
        prediction_values.append(prediction_value)

    return prediction_values
