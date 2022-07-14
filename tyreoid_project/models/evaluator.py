import math
from random import randrange
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist


class Evaluator:

    number_of_folds = 10
    fold_size = 1
    number_of_records_in_training_set = 0

    k_min = 1
    # only 80% of training data will be using as training data for each iterration of cross validation
    k_max = math.sqrt(number_of_records_in_training_set * 0.8)

    def euclidean_distance(self, x, y):

        distance = 0.0
        for i in range(len(x)-1):
            distance += (x[i]-y[i])**2

        return math.sqrt(distance)

    def get_neighbours(self, train_set, test_entry, number_of_neighbors):

        # find distance from every neighbour in train_set, and then sort them by distance ASC
        # get first number_of_neighbours from the top
        list_of_neighbours = list()
        distances = list()

        # distance will be calculated with Euclidean distance formula
        for train_entry in train_set:
            distance_from_neighbour = self.euclidean_distance(train_entry, test_entry)
            distances.append((train_entry, distance_from_neighbour))

        distances.sort(key=lambda d: d[1])

        for i in range(number_of_neighbors):
            list_of_neighbours.append(distances[i][0])

        return list_of_neighbours

    def predict_classification(self, train_set, test_row, number_of_neighbors):

        list_of_neighbours = self.get_neighbours(train_set, test_row, number_of_neighbors)
        target_neighbours_values = [entry[-1] for entry in list_of_neighbours]
        prediction_value = max(set(target_neighbours_values), key=target_neighbours_values.count)

        return prediction_value

    def k_nearest_neighbors(self, train_set, test_set, number_of_neighbors):

         prediction_values = list()
         for entry in test_set:
             prediction_value = self.predict_classification(train_set, entry, number_of_neighbors)
             prediction_values.append(prediction_value)

         print('Prediction values: ')
         print(prediction_values)

         return prediction_values
        
    def accuracy_metric(self, actual_values, prediction_values):

        number_of_correct_predictions = 0
        for i in range(len(actual_values)):
            if actual_values[i] == prediction_values[i]:
                number_of_correct_predictions += 1
            
        return (number_of_correct_predictions / float(len(actual_values))) * 100

    def cross_validation_split(self, dataset):

        #folds will be list of elements, where every element is list of entries in training dataset
        folds = list()
        dataset_tmp = list(dataset.values.tolist())
        self.fold_size = int(len(dataset) / self.number_of_folds)

        for _ in range(self.number_of_folds):
            fold = list()
            for _ in range(self.fold_size-1):
                index = randrange(0, len(dataset_tmp))
                fold.append(dataset_tmp.pop(index))
            folds.append(fold)

        return folds

    def evaluate_algorithm(self, dataset, algorithm, *args):

        scores = list()
        folds = self.cross_validation_split(dataset)

        for idx, fold in enumerate(folds):
            train_folds = list(folds)
            train_folds.remove(fold)
            train_folds = sum(train_folds, [])

            test_fold = list()
            for entry_idx, entry in enumerate(fold):
                entry_tmp = list(entry)
                test_fold.append(entry_tmp)
                entry_tmp[-1] = None #(?) should we do this one line before append?

            print('Calling algorithm for test fold with index: ', idx)
            prediction_values = algorithm(train_folds, test_fold, *args)
            acutal_values = [entry[-1] for entry in fold]
            accuracy_metric = self.accuracy_metric(acutal_values, prediction_values)
            scores.append(accuracy_metric)

        return scores

    def evaluate(self, dataset):

        # 1. KNN algorithm
        # TODO: Should iterrate over k from k_min to k_max, and find best mean of scores
        # for now we put value 10
        print(" --> START of evaluation of KNN algorithm <-- \n")

        self.number_of_records_in_training_set = len(dataset)
        file = open('resources/k_indexes.txt', 'w')

        self.k_max = math.sqrt((self.number_of_records_in_training_set - self.fold_size) * 0.8)
        
        if (self.k_max % 2) == 0:
            self.k_max = self.k_max + 1

        for k_index in range(self.k_min, int(self.k_max)):
            print('K index is: ', k_index)
            scores = self.evaluate_algorithm(dataset, self.k_nearest_neighbors, k_index)
            #print('Scores: %s' % scores)
            #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            #print(" --> END of evaluation of KNN algorithm <-- \n")
            file.write(str(sum(scores) / float(len(scores))))

        file.close()

