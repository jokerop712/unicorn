import requests


url = "http://127.0.0.1:5000/predict"


# Example input JSON
data = {
    "Avg. Session Length": [34.49726773, 31.92627203, 33.00091476, 34.30555663],
    "Time on App": [12.65565115, 11.10946073, 11.33027806, 13.71751367],
    "Time on Website": [50.57766802, 80.26895887, 37.11059744, 36.72128268],
    "Length of Membership": [1.082620633, 2.664034182, 4.104543202, 3.120178783]
}


# Send POST request
response = requests.post(url, json=data)


# Handle response
if response.status_code == 200:
    print("Prediction from API:", response.json())
else:
    print(
        "API Request Failed:",
        response.status_code,
        response.text
    )
