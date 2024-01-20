from django.urls import path
from .views import receive_form_data

urlpatterns = [
    path('receiveFormData/', receive_form_data, name='receive_form_data'),
]