from django.conf.urls import url
from django.urls import path

from . import views

#URLConf
urlpatterns = [
    url(r'^$', views.home),
    path('hello/', views.say_hello),
    path('home/', views.home),
    path('see_prediction_results', views.predict)
]