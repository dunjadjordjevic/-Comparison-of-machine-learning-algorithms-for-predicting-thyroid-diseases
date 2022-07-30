from django.urls import path
from . import views

#URLConf
urlpatterns = [
    path('hello/', views.say_hello),
    path('home/', views.home),
    path('home/see_prediction_results', views.predict)
]