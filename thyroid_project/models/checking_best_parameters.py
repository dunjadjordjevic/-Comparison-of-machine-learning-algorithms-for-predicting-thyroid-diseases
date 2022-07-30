
# Check best parameters for Random forest algorithm
'''
    number_of_trees,
    number_of_data_in_bootstrap_dataset,
    number_of_features,
    max_depth
'''

'''
print('\nCalculating best parameters for Random forest algorithm...')

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


rfc = RandomForestClassifier()
parameters = {
    "bootstrap": [True],
    "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    "n_estimators": [1, 5, 10, 20, 50],
    "max_depth": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

}
cv = GridSearchCV(rfc, parameters, cv=10)
cv.fit(x_train, y_train.values.ravel())
#print(cv.best_params_)
display(cv)
'''

'''
print('\nCalculating best parameters for KNN algorithm...')

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


knn_classifier = KNeighborsClassifier()
k_range = list(range(1, 115, 2))
parameters = {
    "n_neighbors": k_range

}
cv = GridSearchCV(knn_classifier, parameters, cv=10)
cv.fit(x_train, y_train.values.ravel())
display(cv)
'''