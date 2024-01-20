# Decision maker backend

Installation
In order to successfully install this microservice, you need to follow these instructions:

1. Execute the following commands to create and set up your virtual environment:

```python
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
```

#  Run the project
To run this microservice, always use these commands :

```python
source venv/bin/activate
cd backend
python manage.py runserver 8001
```


#  Postman Testing

If you wish to test the APIs using Postman, follow the instructions below:

1- Get the information from the frontend, use the url  "http://127.0.0.1:8001/prediction/getInformation".

2- To send the result of the prediction, use  the URL "http://127.0.0.1:8001/prediction/result".

# Frontend link

```python
https://github.com/oumaima1115/Decision-maker
```
