import requests
import time
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:5000/predict"
API_KEY = "my_secret_key"

def simulate_monitoring():
    response_times = []
    
    for i in range(100):
        start_time = time.time()
        response = requests.post(API_URL, json={'features': [5.1, 3.5, 1.4, 0.2]}, headers={'x-api-key': API_KEY})
        response_time = time.time() - start_time
        response_times.append(response_time)
    
    plt.plot(response_times)
    plt.xlabel('Iteration')
    plt.ylabel('Response Time (seconds)')
    plt.title('API Response Time vs Iteration')
    plt.show()

if __name__ == '__main__':
    simulate_monitoring()



    