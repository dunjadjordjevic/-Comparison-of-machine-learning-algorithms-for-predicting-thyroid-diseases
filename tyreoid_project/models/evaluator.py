from random import randrange
from tyreoid_project.models.algorithms.decision_tree import *
from tyreoid_project.models.algorithms.k_nearest_neighbours import *
from tyreoid_project.models.algorithms.random_forest import *

class Evaluator:

    number_of_folds = 10
    fold_size = 1
    number_of_records_in_training_set = 0

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

    #### Wrappers for calling algorithm implementation methods ###

    def k_nearest_neigbours_call(self, train_set, test_set, columns, number_of_neighbors):

        return k_nearest_neighbors(train_set, test_set, columns, number_of_neighbors)

    def decision_tree_algorithm_call(self, train_set, test_set, columns, min_sample, max_depth):

        return classification_decision_tree(train_set, test_set, columns, min_sample, max_depth)

    def random_forest_algorithm_call(self,
                                     train_set,
                                     test_set,
                                     columns,
                                     number_of_trees,
                                     number_of_data_in_bootstrap_dataset,
                                     number_of_features,
                                     max_depth):
        return random_forest(train_set,
                             test_set,
                             columns,
                             number_of_trees,
                             number_of_data_in_bootstrap_dataset,
                             number_of_features,
                             max_depth)

    def evaluate_algorithm(self, dataset, columns, algorithm, *args):

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
                entry_tmp[-1] = None

            print('Calling algorithm for test fold with index: ', idx)
            prediction_values = algorithm(train_folds, test_fold, columns, *args)
            actual_values = [entry[-1] for entry in fold]
            accuracy_metric = self.accuracy_metric(actual_values, prediction_values)
            scores.append(accuracy_metric)

        return scores

    def evaluate(self, dataset, columns):

        # 1. KNN algorithm
        # TODO: Check what are the best parameters for this algorithm
        # Should iterrate over k from k_min to k_max, and find best mean of scores



        self.number_of_records_in_training_set = len(dataset)
        #file = open('resources/k_indexes.txt', 'w')

        '''
        self.k_min = 1
        self.k_max = math.sqrt((self.number_of_records_in_training_set - self.fold_size) * 0.8)

        if (self.k_max % 2) == 0:
            self.k_max = self.k_max + 1


        print(" --> START of evaluation of KNN algorithm <-- \n")
        for k_index in range(self.k_min, int(self.k_max)):
            print('K index is: ', k_index)
            scores = self.evaluate_algorithm(dataset, columns, self.k_nearest_neigbours_call, k_index)
            print('Scores: %s' % scores)
            print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            print(" --> END of evaluation of KNN algorithm <-- \n")
            #file.write(str(sum(scores) / float(len(scores))))

        #file.close()
        
        # KNN algorithm, k = 10
        #scores = self.evaluate_algorithm(dataset, self.k_nearest_neighbors, 10)
        
        '''

        '''
        # Classification decision tree algorithm
        # TODO: Check what are the best parameters for this algorithm
        min_samples = 2
        max_depth = 5

        print(" --> START of evaluation of classification decision tree algorithm <-- \n")
        dataset['target'] = pd.to_numeric(dataset['target'])
        scores = self.evaluate_algorithm(dataset, columns, self.decision_tree_algorithm_call, min_samples, max_depth)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print(" --> END of evaluation of classification decision tree algorithm <-- \n")
        
        '''

        # Classification random forest algorithm
        # TODO: Check how many decision trees to use for this algorithm

        number_of_trees = 50
        number_of_data_in_bootstrap_dataset = 400
        number_of_features = 800
        max_depth = 12

        print(" --> START of evaluation of random forest algorithm <-- \n")
        dataset['target'] = pd.to_numeric(dataset['target'])
        scores = self.evaluate_algorithm(dataset,
                                         columns,
                                         self.random_forest_algorithm_call,
                                         number_of_trees,
                                         number_of_data_in_bootstrap_dataset,
                                         number_of_features,
                                         max_depth)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print(" --> END of evaluation of random forest algorithm <-- \n")


