import requests

url = 'http://localhost:9696/predict'

patient= {  "gender":"Male",
            "age":70,
            "hypertension":1,
            "heart_disease":1,
            "smoking_history":"former",
            "bmi":40,
            "HbA1c_level":8,
            "blood_glucose_level":200}


response = requests.post(url, json=patient).json()
print(response)

if response['diabetes'] == True:
    print('Sending an appointment email')
else:
    print('NOT sending an appointment email')