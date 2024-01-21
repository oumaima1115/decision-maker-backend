from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from .predict_linear_regression import perform_prediction_and_visualization
from django.conf import settings
import json

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

            results = perform_prediction_and_visualization(csv_file_path, target_variable)

            if results is None or not all(key in results for key in ['predictions', 'residuals', 'download_links']):
                return JsonResponse({'status': 'error', 'message': 'Unexpected results format'})

            predictions = results.get('predictions', [])
            residuals = results.get('residuals', [])
            download_links = results.get('download_links', {})

            images_folder = os.path.join(settings.STATIC, 'images')

            image_urls = []
            for filename in os.listdir(images_folder):
                file_path = os.path.join(images_folder, filename)
                if os.path.isfile(file_path) and filename.endswith(".png"):
                    image_url = f"{settings.STATIC_URL}images/{filename}"
                    image_urls.append(image_url)
            print(image_urls)
            return JsonResponse({
                'status': 'success',
                'message': 'Form data received successfully',
                'predictions': predictions,
                'residuals': residuals,
                'download_links': download_links,
                'image_urls': image_urls,
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
