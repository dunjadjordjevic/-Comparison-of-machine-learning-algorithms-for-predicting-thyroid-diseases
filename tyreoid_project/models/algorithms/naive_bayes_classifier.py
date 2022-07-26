from math import sqrt, pi, exp
#from statistics import mean, stdev


def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

# Create dictionary where
# key is value of target class
# value of each key is list of entries which target class is matching the key of dict
def divide_by_class(train_set):

    groups_by_class = dict()

    for i in range(len(train_set)):

        entry = train_set[i]
        target_class = entry[-1]

        if target_class not in groups_by_class:
            groups_by_class[target_class] = list()

        groups_by_class[target_class].append(entry)

    return groups_by_class


# Input - list of entries that belongs to some class
# Output - calculates list of metrics data for that group of entries
def summarize_dataset(entries):

    # zip(*entries) returns iterable and do parallel iterations over iterable collection
    data_metrics = [(mean(column), stdev(column), len(column)) for column in zip(*entries)]

    return data_metrics

# Divide train dataset by class in tree groups and calculate for each row following information:
# - mean - average value of list
# - stdev
# - number of data in each column
def summarize_by_class(train_set):

    groups_by_class = divide_by_class(train_set)
    summaries = dict()

    for target_class, entries in groups_by_class.items():
        summaries[target_class] = summarize_dataset(entries)

    return summaries

# Calculate probability by Gaussian formula
def calculate_probability(x, mean, stdev):

    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    probability = (1 / (sqrt(2 * pi) * stdev)) * exponent

    return probability

# Creates dictionary where key is target class, and value is probability for that class
def calculate_class_probabilities(summarize, entry):

    probabilities = dict()
    total_rows = sum([summarize[target_class][0][2] for target_class in summarize])

    for class_value, class_summaries in summarize.items():
        probabilities[class_value] = summarize[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            if (mean != 0 and stdev != 0):
                probabilities[class_value] *= calculate_probability(entry[i], mean, stdev)

    return probabilities

# For given entry, this function should give as a result class
# depending on summarize metrics information
def predict_class(summarize, entry):

    predicted_class = None
    best_probability = -1

    probabilities_dict = calculate_class_probabilities(summarize, entry)

    for probability_class, probability in probabilities_dict.items():
        if best_probability is None or probability > best_probability:
            best_probability = probability
            predicted_class = probability_class

    return predicted_class

def naive_bayes_classifier(self, train_set, test_set, columns):

    for entry in train_set:
        entry[-1] = int(entry[-1])

    prediction_values = list()
    summarize = summarize_by_class(train_set)

    for entry in test_set:
        prediction_value = predict_class(summarize, entry)
        prediction_values.append(prediction_value)

    print('Prediction values: ')
    print(prediction_values)

    return prediction_values