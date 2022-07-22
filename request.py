import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'SepalLengthCm':5.1, 'SepalWidthCm':3.5, 'PetalLengthCm':1.4,'PetalWidthCm':0.2})

print(r.json())