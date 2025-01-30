import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify, abort
import time
import matplotlib.pyplot as plt

API_KEY = 'my_secret_key'


# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    # Load dataset (using the Iris dataset as an example)
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # Split the data into features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Step 2: Train the machine learning model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 3: Evaluate the machine learning model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

# Step 4: Deploy the model via an API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    #enforce API key
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        print("Unauthorized")
        abort(401)  # Unauthorized

    data = request.get_json(force=True)

    #Enfore data validation
    if not validate_input(data):
        abort(400)  # Bad Request

    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

# Step 5: Input validation for security purposes and correct processing
def validate_input(data):
    #Ensuring data is a dictionary
    if not isinstance(data, dict):
        return False
    #Ensuring the data contains features
    if 'features' not in data:
        return False
    #Ensuring features is a list
    if not isinstance(data['features'], list):
        return False
    #Ensuring all features are numeric data types
    if not all(isinstance(x, (int, float)) for x in data['features']):
        return False
    return True

# Step 6: Simulate monitoring in production
def simulate_monitoring():
    response_times = []
    
    for i in range(200):
        start_time = time.time()
        response = app.test_client().post('/predict', 
                                          json={'features': [5.1, 3.5, 1.4, 0.2]}, 
                                          headers={'x-api-key' :'my_secret_key'})
        response_time = time.time() - start_time
        response_times.append(response_time)
    
    plt.plot(response_times)
    plt.xlabel('Iteration')
    plt.ylabel('Response Time (seconds)')
    plt.title('API Response Time vs Iteration')
    plt.show()

if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Run the Flask app (API deployment)
    app.run(debug=True)

# Note: To simulate monitoring in production, run the simulate_monitoring function in a separate script or after deploying the API.