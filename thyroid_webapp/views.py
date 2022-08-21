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

            prediction_dataset["predicted_values"] = [str(x) for x in prediction_dataset["predicted_values"]]
            prediction_dataset["actual_values"] = [str(x) for x in prediction_dataset["actual_values"]]

            for i in prediction_dataset.index:

                if (prediction_dataset.at[i, "predicted_values"] == '0'):
                    prediction_dataset.at[i, "predicted_values"] = '-'
                elif (prediction_dataset.at[i, "predicted_values"] == '1'):
                    prediction_dataset.at[i, "predicted_values"] = 'hyperthyroid'
                elif (prediction_dataset.at[i, "predicted_values"] == '2'):
                    prediction_dataset.at[i, "predicted_values"] = 'T3 toxic'
                elif (prediction_dataset.at[i, "predicted_values"] == '3'):
                    prediction_dataset.at[i, "predicted_values"] = 'toxic goitre'
                elif (prediction_dataset.at[i, "predicted_values"] == '4'):
                    prediction_dataset.at[i, "predicted_values"] = 'secondary toxic'
                elif (prediction_dataset.at[i, "predicted_values"] == '5'):
                    prediction_dataset.at[i, "predicted_values"] = 'hypothyroid'
                elif (prediction_dataset.at[i, "predicted_values"] == '6'):
                    prediction_dataset.at[i, "predicted_values"] = 'primary hypothyroid'
                elif (prediction_dataset.at[i, "predicted_values"] == '7'):
                    prediction_dataset.at[i, "predicted_values"] = 'compensated hypothyroid'
                elif (prediction_dataset.at[i, "predicted_values"] == '8'):
                    prediction_dataset.at[i, "predicted_values"] = 'secondary hypothyroid'
                elif (prediction_dataset.at[i, "predicted_values"] == '9'):
                    prediction_dataset.at[i, "predicted_values"] = 'increased binding protein'
                elif (prediction_dataset.at[i, "predicted_values"] == '10'):
                    prediction_dataset.at[i, "predicted_values"] = 'decreased binding protein'
                elif (prediction_dataset.at[i, "predicted_values"] == '11'):
                    prediction_dataset.at[i, "predicted_values"] = 'concurrent non-thyroidal illness'
                elif (prediction_dataset.at[i, "predicted_values"] == '12'):
                    prediction_dataset.at[i, "predicted_values"] = 'consistent with replacement therapy'
                elif (prediction_dataset.at[i, "predicted_values"] == '13'):
                    prediction_dataset.at[i, "predicted_values"] = 'underreplaced'
                elif (prediction_dataset.at[i, "predicted_values"] == '14'):
                    prediction_dataset.at[i, "predicted_values"] = 'overreplaced'
                elif (prediction_dataset.at[i, "predicted_values"] == '15'):
                    prediction_dataset.at[i, "predicted_values"] = 'antithyroid drugs'
                elif (prediction_dataset.at[i, "predicted_values"] == '16'):
                    prediction_dataset.at[i, "predicted_values"] = 'I131 treatment'
                elif (prediction_dataset.at[i, "predicted_values"] == '17'):
                    prediction_dataset.at[i, "predicted_values"] = 'surgery'
                else:
                    prediction_dataset.at[i, "predicted_values"] = 'miscellaneous'

                if (prediction_dataset.at[i, "actual_values"] == '0'):
                    prediction_dataset.at[i, "actual_values"] = '-'
                elif (prediction_dataset.at[i, "actual_values"] == '1'):
                    prediction_dataset.at[i, "actual_values"] = 'hyperthyroid'
                elif (prediction_dataset.at[i, "actual_values"] == '2'):
                    prediction_dataset.at[i, "actual_values"] = 'T3 toxic'
                elif (prediction_dataset.at[i, "actual_values"] == '3'):
                    prediction_dataset.at[i, "actual_values"] = 'toxic goitre'
                elif (prediction_dataset.at[i, "actual_values"] == '4'):
                    prediction_dataset.at[i, "actual_values"] = 'secondary toxic'
                elif (prediction_dataset.at[i, "actual_values"] == '5'):
                    prediction_dataset.at[i, "actual_values"] = 'hypothyroid'
                elif (prediction_dataset.at[i, "actual_values"] == '6'):
                    prediction_dataset.at[i, "actual_values"] = 'primary hypothyroid'
                elif (prediction_dataset.at[i, "actual_values"] == '7'):
                    prediction_dataset.at[i, "actual_values"] = 'compensated hypothyroid'
                elif (prediction_dataset.at[i, "actual_values"] == '8'):
                    prediction_dataset.at[i, "actual_values"] = 'secondary hypothyroid'
                elif (prediction_dataset.at[i, "actual_values"] == '9'):
                    prediction_dataset.at[i, "actual_values"] = 'increased binding protein'
                elif (prediction_dataset.at[i, "actual_values"] == '10'):
                    prediction_dataset.at[i, "actual_values"] = 'decreased binding protein'
                elif (prediction_dataset.at[i, "actual_values"] == '11'):
                    prediction_dataset.at[i, "actual_values"] = 'concurrent non-thyroidal illness'
                elif (prediction_dataset.at[i, "actual_values"] == '12'):
                    prediction_dataset.at[i, "actual_values"] = 'consistent with replacement therapy'
                elif (prediction_dataset.at[i, "actual_values"] == '13'):
                    prediction_dataset.at[i, "actual_values"] = 'underreplaced'
                elif (prediction_dataset.at[i, "actual_values"] == '14'):
                    prediction_dataset.at[i, "actual_values"] = 'overreplaced'
                elif (prediction_dataset.at[i, "actual_values"] == '15'):
                    prediction_dataset.at[i, "actual_values"] = 'antithyroid drugs'
                elif (prediction_dataset.at[i, "actual_values"] == '16'):
                    prediction_dataset.at[i, "actual_values"] = 'I131 treatment'
                elif (prediction_dataset.at[i, "actual_values"] == '17'):
                    prediction_dataset.at[i, "actual_values"] = 'surgery'
                else:
                    prediction_dataset.at[i, "actual_values"] = 'miscellaneous'

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
