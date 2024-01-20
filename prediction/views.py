from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from .utils import make_prediction
from .predict_linear_regression import perform_prediction_and_visualization
import matplotlib.pyplot as plt
import seaborn as sns

@csrf_exempt
def receive_form_data(request):
    print('Received request to receive_form_data')
    if request.method == 'POST':
        try:
            target_variable = request.POST.get('targetVariable')
            csv_file = request.FILES.get('csvFile')
            csv_file_path = 'prediction/uploaded_data.csv'

            with open(csv_file_path, 'wb') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)

            df = pd.read_csv(csv_file_path, sep=';')
            
            if target_variable not in df.columns:
                return JsonResponse({'status': 'error', 'message': f'Target variable "{target_variable}" not found in the dataset'})

            predictions, residuals, download_links, scatter_plot_base64 = perform_prediction_and_visualization(csv_file_path, target_variable)
            print(download_links)

            return JsonResponse({
                'status': 'success',
                'message': 'Form data received successfully',
                'scatter_plot_base64': scatter_plot_base64,
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
