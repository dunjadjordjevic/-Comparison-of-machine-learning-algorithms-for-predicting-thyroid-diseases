from django.db import models
from django import forms

ALGORITHMS = {
    'k_nearest_neighbours': 'K najbližih suseda',
    'decision_tree_algorithm' : 'Stabla odlučivanja',
    'random_forest_algorithm' : 'Slučajne šume',
    'naive_bayes_classifier_algorithm' : 'Naivni Bajesov klasifikator'
}

class ModelFormWithFileField(forms.Form):

    datasetTestFile = models.TextField()
    nameOfAlgorithm = models.TextField()

