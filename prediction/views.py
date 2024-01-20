from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def receive_form_data(request):
    print('Received request to receive_form_data')
    if request.method == 'POST':
        try:
            target_variable = request.POST.get('targetVariable')
            show_diagram = request.POST.get('showDiagram')
            csv_file = request.FILES.get('csvFile')

            print('target_variable:', target_variable)
            print('show_diagram:', show_diagram)
            print('csv_file:', csv_file)

            # Process the form data as needed

            return JsonResponse({'status': 'success', 'message': 'Form data received successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
