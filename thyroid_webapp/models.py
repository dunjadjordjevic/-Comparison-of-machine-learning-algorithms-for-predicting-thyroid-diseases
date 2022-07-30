from django.db import models
from django import forms

class ModelFormWithFileField(forms.Form):

    datasetTestFile = models.TextField()
    nameOfAlgorithm = models.TextField()

