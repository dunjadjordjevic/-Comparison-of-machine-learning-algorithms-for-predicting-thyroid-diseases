import pandas as pd
from tyreoid_project.models.algorithms.decision_tree import classify_entry

def make_predictions(df, tree):

    if len(df) != 0:
        predictions = df.apply(classify_entry, args=(tree,), axis=1)
    else:
        predictions = pd.Series()

    return predictions

def filter_df(df, question):

    feature, comparison_operator, value = question.split()

    # continuous feature
    if comparison_operator == "<=":
        df_yes = df[df[feature] <= float(value)]
        df_no = df[df[feature] > float(value)]

    # categorical feature
    else:
        df_yes = df[df[feature].astype(str) == value]
        df_no = df[df[feature].astype(str) != value]

    return df_yes, df_no


def determine_leaf(df_train):
    return df_train.target.value_counts().index[0] #first element is the most frequently-occurring element


def determine_errors(df_val, tree):

    predictions = make_predictions(df_val, tree)
    actual_values = df_val.target

    return sum(predictions != actual_values)


def pruning_result(tree, df_train, df_val):

    leaf = determine_leaf(df_train)
    errors_leaf = determine_errors(df_val, leaf)
    errors_decision_node = determine_errors(df_val, tree)

    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree


def post_pruning(tree, df_train, df_val, columns):

    df_train = pd.DataFrame(data=df_train, columns=columns)
    df_val = pd.DataFrame(data=df_val, columns=columns)

    # for each decision node, check if we should keep it or not
    # with recursive function traverse the tree

    question = list(tree.keys())[0]
    yes_answer, no_answer = tree[question]

    # base case
    if not isinstance(yes_answer, dict) and not isinstance(no_answer, dict):
        return pruning_result(tree, df_train, df_val)
    else:
        df_train_yes, df_train_no = filter_df(df_train, question)
        df_val_yes, df_val_no = filter_df(df_val, question)

        if isinstance(yes_answer, dict):
            yes_answer = post_pruning(yes_answer, df_train_yes, df_val_yes, columns=columns)

        if isinstance(no_answer, dict):
            no_answer = post_pruning(no_answer, df_train_no, df_val_no, columns=columns)

        tree = {question: [yes_answer, no_answer]}

        return pruning_result(tree, df_train, df_val)