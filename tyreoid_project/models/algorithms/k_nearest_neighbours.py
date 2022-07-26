import math

def euclidean_distance(x, y, w):

    distance = 0.0
    for i in range(len(x) - 1):
        distance += (w[i]*(x[i] - y[i])) ** 2

    return math.sqrt(distance)


def get_neighbours(train_set, test_entry, number_of_neighbors, w):

    # find distance from every neighbour in train_set, and then sort them by distance ASC
    # get first number_of_neighbours from the top
    list_of_neighbours = list()
    distances = list()

    # distance will be calculated with Euclidean distance formula
    for train_entry in train_set:
        distance_from_neighbour = euclidean_distance(train_entry, test_entry, w)
        distances.append((train_entry, distance_from_neighbour))

    distances.sort(key=lambda d: d[1])

    for i in range(number_of_neighbors):
        list_of_neighbours.append(distances[i][0])

    return list_of_neighbours


def predict_classification(train_set, test_row, number_of_neighbors, w):

    list_of_neighbours = get_neighbours(train_set, test_row, number_of_neighbors, w)
    target_neighbours_values = [entry[-1] for entry in list_of_neighbours]
    prediction_value = max(set(target_neighbours_values), key=target_neighbours_values.count)

    return prediction_value

def k_nearest_neighbors(train_set, test_set, columns, number_of_neighbors, w):

    prediction_values = list()
    for entry in test_set:
        prediction_value = predict_classification(train_set, entry, number_of_neighbors, w)
        prediction_values.append(prediction_value)

    print('Prediction values: ')
    print(prediction_values)

    return prediction_values
