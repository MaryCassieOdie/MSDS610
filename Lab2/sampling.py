import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

def load_and_prep_data():
    # Load dataset
    iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # Split the data into features and target
    iris_features = iris_data.drop('class', axis=1)
    iris_target = iris_data['class']
    return iris_data, iris_features, iris_target

def strat_split(iris_data, iris_features, iris_target):
    # Stratified Shuffle Split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in strat_split.split(iris_features, iris_target):
        strat_train_set = iris_data.loc[train_index]
        strat_test_set = iris_data.loc[test_index]

    return strat_train_set, strat_test_set


def random_samp(iris_data): 
    return train_test_split(iris_data, test_size=0.2, random_state=42)


def systematic_step_samp(iris_data):
    step = 5
    # Systematic Sampling

    indices = list(range(0, len(iris_data), step))
    return iris_data.iloc[indices]


# Cluster Sampling function
def cluster_sampling(data):
    n_clusters = 3
    clusters = np.array_split(data, n_clusters)
    sampled_clusters = np.random.choice(range(n_clusters), size=n_clusters//2, replace=False)
    return pd.concat([clusters[i] for i in sampled_clusters])

def test_class_distribution(train_set,test_set):
    # Verify class distribution
    train_class_distribution = train_set['class'].value_counts(normalize=True)
    test_class_distribution = test_set['class'].value_counts(normalize=True)

    print("Training set class distribution:\n", train_class_distribution)
    print("Testing set class distribution:\n", test_class_distribution)


if __name__ == '__main__':

    iris_data, iris_features, iris_target = load_and_prep_data()

    #load and prep data stratified split
    strat_train_set, strat_test_set = strat_split(iris_data, iris_features, iris_target)
    #test class distribution - Stratified Split
    print("Stratified Split")
    test_class_distribution(strat_train_set,strat_test_set)

    #Test RAndom sampling
    random_train, random_test = random_samp(iris_data)
    print("Random Sampling")
    test_class_distribution(random_train, random_test)

    #setup and test Systematic Step
    systematic_train = systematic_step_samp(random_train)
    systematic_test = systematic_step_samp(random_test)
    print("Systematic Step")
    test_class_distribution(systematic_train, systematic_test)

        #setup and test Systematic Step
    systematic_train = cluster_sampling(random_train)
    systematic_test = cluster_sampling(random_test)
    print("Cluster Sampling")
    test_class_distribution(systematic_train, systematic_test)

    # Display the results
    print("Training set:\n", strat_train_set)
    print("Testing set:\n", strat_test_set)