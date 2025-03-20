from django.urls import path
from .views import GenerateSummary

urlpatterns = [
    path('getsummary/', GenerateSummary.as_view()),
]