import json

from django.shortcuts import render
from django.http import HttpResponse
from .models import ModelFormWithFileField, ALGORITHMS
from thyroid_project.main import predictDataWithMetrics, mainMethod, prepareThyroidDataset

def say_hello(request):
    return render(request, 'hello.html', {'name': 'Duki'})

def home(request):
    return render(request, 'home.html', {'algorithms': ALGORITHMS})

def predict(request):

    print('Prediction for new dataset has started...')

    if request.method == 'POST':

        form = ModelFormWithFileField(request.POST, request.FILES['datasetTestFile'].name)

        if form.is_valid():

            fileForPrediction = request.FILES['datasetTestFile']
            thyroidDataset, columns, x_train, x_test, y_train, y_test = prepareThyroidDataset()
            thyroidDataset, columns, x_train, x_test, y_train, y_test = mainMethod(thyroidDataset, columns, x_train,
                                                                                   x_test, y_train, y_test,
                                                                                   evaluate=False)
            datasetWithPrediction, columns, accuracy_metric, precision, recall, f1score, sensitivity = \
                predictDataWithMetrics(fileForPrediction.name[:-4], request.POST['nameOfAlgorithm'], columns, x_train, y_train)

            prediction_dataset = datasetWithPrediction
            prediction_dataset['TSH'] = prediction_dataset['TSH'].map('{:,.2f}'.format)
            prediction_dataset['T3'] = prediction_dataset['T3'].map('{:,.2f}'.format)
            prediction_dataset['TT4'] = prediction_dataset['TT4'].map('{:,.2f}'.format)
            prediction_dataset['T4U'] = prediction_dataset['T4U'].map('{:,.2f}'.format)
            prediction_dataset['FTI'] = prediction_dataset['FTI'].map('{:,.2f}'.format)

            json_records = prediction_dataset.reset_index().to_json(orient='records')
            data = []
            data = json.loads(json_records)

            request.session['data'] = data
            request.session['accuracyMetric'] = '{:,.3f}'.format(accuracy_metric)
            return render(request, 'home.html',  {'algorithms': ALGORITHMS, 'data': data, 'accuracyMetric': accuracy_metric})
        else:

            print('Form with data is not valid - imported data are not valid. Please, try again with changed data.')
            return HttpResponse(status=204)

    return render(request, 'home.html', {'algorithms': ALGORITHMS})
