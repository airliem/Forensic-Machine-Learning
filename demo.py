#code to set up local host to run predictions
import requests

resp = requests.post('http://localhost:5000/predict',
                    files={"file": open('/home/melissa/Documents/Extra_Modified_Dataset/T1/T1_2.jpg', 'rb')})

print(resp.json())
