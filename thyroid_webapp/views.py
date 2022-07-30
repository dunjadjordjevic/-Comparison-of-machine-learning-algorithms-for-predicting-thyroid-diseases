import imblearn
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import ModelFormWithFileField
from thyroid_project.main import predictDataWithMetrics

def say_hello(request):
    #return HttpResponse('Hello world!')
    return render(request, 'hello.html', {'name': 'Duki'})

def home(request):
    return render(request, 'home.html')

def predict(request):
    print('Predikcija nad novim dataset-om je započeta...')

    if request.method == 'POST':

        form = ModelFormWithFileField(request.POST, request.FILES['datasetTestFile'].name)

        if form.is_valid():
            fileForPrediction = request.FILES['datasetTestFile']
            datasetWithPrediction, columns, accuracy_metric, precision, recall, f1score = \
                predictDataWithMetrics(fileForPrediction.name[:-4], request.POST['nameOfAlgorithm'])

            return HttpResponseRedirect('/home/#predict_page')
    else:
        print('Forma nije validna - uneti podaci nisu ispravni. Pokusajte ponovo.')
        return HttpResponse(status=204)

    #TDOO: Update home.html to be same page
    return render(request, 'home.html', {'form': form})