import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load dataset
iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Split the data into features and target
iris_features = iris_data.drop('class', axis=1)
iris_target = iris_data['class']

# Stratified Shuffle Split
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strat_split.split(iris_features, iris_target):
    strat_train_set = iris_data.loc[train_index]
    strat_test_set = iris_data.loc[test_index]

# Display the results
print("Training set:\n", strat_train_set)
print("Testing set:\n", strat_test_set)