from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def say_hello(request):
    #return HttpResponse('Hello world!')
    return render(request, 'hello.html', {'name': 'Duki'})

def home(request):
    return render(request, 'home.html')