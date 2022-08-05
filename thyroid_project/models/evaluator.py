import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from random import randrange
from thyroid_project.models.algorithms.decision_tree import *
from thyroid_project.models.algorithms.k_nearest_neighbours import *
from thyroid_project.models.algorithms.random_forest import *
from thyroid_project.models.algorithms.naive_bayes_classifier import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_metrics(scores, precision, recall, f1score):

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    print('Precision: %.3f%%' % (sum(precision) / float(len(precision))))
    print('Recall: %.3f%%' % (sum(recall) / float(len(recall))))
    print('F1 score: %.3f%%' % (sum(f1score) / float(len(f1score))))


class Evaluator:

    number_of_folds = 10
    fold_size = 1
    number_of_records_in_training_set = 0

    def accuracy_metric(self, actual_values, prediction_values):

        actual_values = list(map(int, actual_values))
        prediction_values = list(map(int, prediction_values))

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

    ### Function for calculation weights for attributes ####
    def regularization_coeffients(self, X, target, lambd=1):

        reg_coefficients = []
        for row in X.transpose().iterrows():
            r = abs(pearsonr(list(row[1]), target)[0]) ** lambd
            reg_coefficients.append(r)

        reg_coefficients = [1 if math.isnan(x) else x for x in reg_coefficients]

        return reg_coefficients


    #### Wrappers for calling algorithm implementation methods ###

    def k_nearest_neigbours_call(self, train_set, test_set, columns, number_of_neighbors, w=[1]*21):

        return k_nearest_neighbors(train_set, test_set, columns, number_of_neighbors, w)

    def decision_tree_algorithm_call(self, train_set, test_set, columns, min_sample, max_depth, metric_function):

        return classification_decision_tree(train_set, test_set, columns, min_sample, max_depth, metric_function)

    def random_forest_algorithm_call(self,
                                     train_set,
                                     test_set,
                                     columns,
                                     number_of_trees,
                                     number_of_data_in_bootstrap_dataset,
                                     number_of_features,
                                     max_depth,
                                     metric_function):
        return random_forest(train_set,
                             test_set,
                             columns,
                             number_of_trees,
                             number_of_data_in_bootstrap_dataset,
                             number_of_features,
                             max_depth,
                             metric_function)

    def naive_bayes_classifier_call(self,  train_set, test_set, columns):

        return naive_bayes_classifier(self, train_set, test_set, columns)

    def evaluate_algorithm(self, dataset, columns, algorithm, *args):

        scores = list()
        precision = list()
        recall = list()
        f1score = list()

        folds = self.cross_validation_split(dataset)

        for idx, fold in enumerate(folds):
            train_folds = list(folds)
            train_folds.remove(fold)
            train_folds = sum(train_folds, [])

            test_fold = list()
            for entry_idx, entry in enumerate(fold):
                entry_tmp = list(entry)
                test_fold.append(entry_tmp)
                #entry_tmp[-1] = None

            print('Calling algorithm for test fold with index: ', idx)
            prediction_values = algorithm(train_folds, test_fold, columns, *args)
            prediction_values = [int(x) for x in prediction_values]
            actual_values = [entry[-1] for entry in fold]
            actual_values = [int(x) for x in actual_values]
            accuracy_metric = accuracy_score(actual_values, prediction_values)
            scores.append(accuracy_metric*100)
            precision.append(precision_score(actual_values, prediction_values, average='weighted')*100)
            recall.append(recall_score(actual_values, prediction_values, average='weighted')*100)
            f1score.append(f1_score(actual_values, prediction_values, average='weighted')*100)

        print(scores)
        return scores, precision, recall, f1score

    def predict_algorithm(self, train_set, test_set, columns, algorithm_name, *args):

        train_set = list(train_set.values.tolist())
        test_set = list(test_set.values.tolist())

        if (algorithm_name == 'k_nearest_neighbours'):
            algorithm = self.k_nearest_neigbours_call
        elif(algorithm_name == 'decision_tree_algorithm'):
            algorithm = self.decision_tree_algorithm_call
        elif (algorithm_name == 'random_forest_algorithm'):
            algorithm = self.random_forest_algorithm_call
        elif (algorithm_name == 'naive_bayes_classifier_algorithm'):
            algorithm = self.naive_bayes_classifier_call


        print('--> Starting with prediction on test set <--')
        prediction_values = algorithm(train_set, test_set, columns, *args)
        prediction_values = [int(x) for x in prediction_values]
        actual_values = [entry[-1] for entry in test_set]
        actual_values = [int(x) for x in actual_values]
        accuracy_metric = accuracy_score(actual_values, prediction_values) * 100
        precision = precision_score(actual_values, prediction_values, average='weighted') * 100
        recall = recall_score(actual_values, prediction_values, average='weighted') * 100
        f1score = f1_score(actual_values, prediction_values, average='weighted') * 100
        print('Mean Accuracy for prediction is : %.3f%%' % accuracy_metric)
        print('Precision for prediction is : %.3f%%' % precision)
        print('Recall for prediction is : %.3f%%' % recall)
        print('F1 score for prediction is : %.3f%%' % f1score)
        print('--> End of prediction on test set <--')

        return prediction_values, accuracy_metric, precision, recall, f1score

    def evaluate(self, dataset, columns):

        '''
        # 1. KNN algorithm
        # Check what are the best parameters for this algorithm
        # Should iterrate over k from k_min to k_max, and find best mean of scores
        precision_values = list()
        recall_values = list()
        f1score_values = list()

        self.number_of_records_in_training_set = len(dataset)

        self.k_min = 3
        self.k_max = math.sqrt((self.number_of_records_in_training_set - self.fold_size) * 0.8)

        if (self.k_max % 2) == 0:
            self.k_max = self.k_max + 1

        k_values = []
        accuracy_values = []

        y_train = dataset['target'].astype(int)
        x_train = dataset.drop('target', axis=1)

        w = self.regularization_coeffients(x_train, y_train.values)
        f = open("thyroid_project/resources/k_values.txt", "w")

        print(" --> START of evaluation of KNN algorithm <-- \n")

        for k_index in range(self.k_min, 131, 2):
            if (k_index % 2) == 0:
                continue

            print('K index is: ', k_index)
            f.write("k: " + str(k_index))
            k_values.append(k_index)
            scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.k_nearest_neigbours_call, k_index, w)
            print_metrics(scores, precision, recall, f1score)
            accuracy_values.append((sum(scores) / float(len(scores))))
            precision_values.append((sum(precision) / float(len(precision))))
            recall_values.append((sum(recall) / float(len(recall))))
            f1score_values.append((sum(f1score) / float(len(f1score))))
            f.write("accuracy: " + str((sum(scores) / float(len(scores)))))
            f.write("\n")
            print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            print(" --> END of evaluation of KNN algorithm <-- \n")

        plt.plot(k_values, accuracy_values, label="Dependency of K and accuracy in KNN algorithm")
        plt.title("Dependency of K and accuracy in KNN algorithm")
        plt.xlabel("Number of neighbours")
        plt.ylabel("Accuracy metric")
        plt.savefig('resources/generated_images/training_evaluation_images/knn_k_accuracy.png')
        #plt.show()
        plt.clf()
        plt.cla()

        plt.plot(k_values, precision_values, label="Dependency of K and precision in KNN algorithm")
        plt.title("Dependency of K and precision in KNN algorithm")
        plt.xlabel("Number of neighbours")
        plt.ylabel("Precision metric")
        plt.savefig('resources/generated_images/training_evaluation_images/knn_k_precision.png')
        plt.clf()
        plt.cla()

        plt.plot(k_values, recall_values, label="Dependency of K and recall in KNN algorithm")
        plt.title("Dependency of K and recall in KNN algorithm")
        plt.xlabel("Number of neighbours")
        plt.ylabel("Recall metric")
        plt.savefig('resources/generated_images/training_evaluation_images/knn_k_recall.png')
        plt.clf()
        plt.cla()

        plt.plot(k_values, f1score_values, label="Dependency of K and F1 score in KNN algorithm")
        plt.title("Dependency of K and F1 score in KNN algorithm")
        plt.xlabel("Number of neighbours")
        plt.ylabel("F1 score metric")
        plt.savefig('resources/generated_images/training_evaluation_images/knn_k_f1score.png')
        plt.clf()
        plt.cla()

        f.close()

        '''

        '''
        # KNN algorithm, k = 109

        y_train = dataset['target'].astype(int)
        x_train = dataset.drop('target', axis=1)
        w = self.regularization_coeffients(x_train, y_train.values)

        print(" --> START of evaluation of KNN algorithm <-- \n")
        scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.k_nearest_neigbours_call, 109, w)
        print_metrics(scores, precision, recall, f1score)
        print(" --> END of evaluation of KNN algorithm <-- \n")
        '''

        '''
        # Classification decision tree algorithm
        min_samples = 2
        max_depth = 5
        metric_function = calculate_entropy

        print(" --> START of evaluation of classification decision tree algorithm <-- \n")
        dataset['target'] = pd.to_numeric(dataset['target'])
        scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.decision_tree_algorithm_call, min_samples, max_depth, metric_function)
        print_metrics(scores, precision, recall, f1score)
        print(" --> END of evaluation of classification decision tree algorithm <-- \n")
        '''

        '''
        # Calculation of the best max_depth parameter for classification decision tree

        min_samples = 2
        max_depth_from = 1
        max_depth_to = len(columns)-1

        max_depth_values = []
        accuracy_gini = []
        accuracy_entropy = []
        dataset['target'] = pd.to_numeric(dataset['target'])

        print(" --> START of evaluation of classification decision tree algorithm <-- \n")

        print(" --> Evaluation for ENTROPY metric <-- \n")
        metric_function = calculate_entropy
        for depth_of_tree in range(max_depth_from, max_depth_to, 2):
            max_depth_values.append(depth_of_tree)
            scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.decision_tree_algorithm_call, min_samples, depth_of_tree, metric_function)
            print_metrics(scores, precision, recall, f1score)
            accuracy_entropy.append(sum(scores) / float(len(scores)))

        print(" --> Evaluation for GINI metric <-- \n")
        metric_function = calculate_gini
        for depth_of_tree in range(max_depth_from, max_depth_to, 2):
            scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.decision_tree_algorithm_call, min_samples, depth_of_tree, metric_function)
            print_metrics(scores, precision, recall, f1score)
            accuracy_gini.append(sum(scores) / float(len(scores)))

        print(" --> END of evaluation of classification decision tree algorithm <-- \n")

        plt.plot(max_depth_values, accuracy_gini)
        plt.plot(max_depth_values, accuracy_entropy)
        plt.xlabel("Maximum depth of a tree")
        plt.ylabel("Accuracy metric")
        plt.savefig('resources/generated_images/training_evaluation_images/dt_max_depth_accuracy.png')
        #plt.show()
        plt.clf()
        plt.cla()

        '''
        '''
        # Classification random forest algorithm

        number_of_trees = 50
        number_of_data_in_bootstrap_dataset = 400
        number_of_features = 800
        max_depth = 12
        metric_function = calculate_entropy

        print(" --> START of evaluation of random forest algorithm <-- \n")
        dataset['target'] = pd.to_numeric(dataset['target'])
        scores, precision, recall, f1score = self.evaluate_algorithm(dataset,
                                         columns,
                                         self.random_forest_algorithm_call,
                                         number_of_trees,
                                         number_of_data_in_bootstrap_dataset,
                                         number_of_features,
                                         max_depth,
                                         metric_function)
        print_metrics(scores, precision, recall, f1score)
        print(" --> END of evaluation of random forest algorithm <-- \n")
        '''

        '''
        # Naive Bayes classifier algorithm
        print(" --> START of evaluation of Naive Bayes classifier algorithm <-- \n")
        scores, precision, recall, f1score = self.evaluate_algorithm(dataset, columns, self.naive_bayes_classifier_call)
        print_metrics(scores, precision, recall, f1score)
        print(" --> END of evaluation of Naive Bayes classifier algorithm <-- \n")
        '''
