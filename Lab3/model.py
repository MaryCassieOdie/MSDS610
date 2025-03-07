import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.utils import resample


def load_and_split_data():
    # load data and separate features/target
    df = pd.read_csv('./winequality-white.csv', header=0)
    features = df.drop(labels='quality', axis=1)
    target = df['quality']

    # First split dataset
    features_train, features_temp, target_train, target_temp = train_test_split(features, target, test_size=0.3, random_state=42)

    # Second split: 50% of the temporary set for validation and 50% for test
    features_val, features_test, target_val, target_test = train_test_split(features_temp, target_temp, test_size=0.5, random_state=42)

    return df, features_train, target_train, features_val, features_test, target_val, target_test

def sim_covariate_shift(test_data):
    # Simulate covariate shift by adding noise to test set
    test_shifted = test_data + np.random.normal(0, 0.5, test_data.shape)
    return test_shifted

def sim_label_shift(data_set):
    #separate existing classes
    class3 = data_set[data_set['quality'] == 3]
    class4 = data_set[data_set['quality'] == 4]
    class5 = data_set[data_set['quality'] == 5]
    class6 = data_set[data_set['quality'] == 6]
    class7 = data_set[data_set['quality'] == 7]
    class8 = data_set[data_set['quality'] == 8]
    class9 = data_set[data_set['quality'] == 9]

    #Determine total number of samples, and desired class distribution
    n_samples = len(data_set)
    p_3 = .05
    p_4 = .2
    p_5 = .2
    p_6 = .2
    p_7 = .2
    p_8 = .1
    p_9 = .05

    #Calculate number of desired samples for each class
    n_class_3 = int(n_samples * p_3)
    n_class_4 = int(n_samples * p_4)
    n_class_5 = int(n_samples * p_5)
    n_class_6 = int(n_samples * p_6)
    n_class_7 = int(n_samples * p_7)
    n_class_8 = int(n_samples * p_8)
    n_class_9 = int(n_samples * p_9)

    #resample based on desired number of records calculated
    class_3_resampled = resample(class3, replace=True, n_samples=n_class_3, random_state=42)
    class_4_resampled = resample(class4, replace=True, n_samples=n_class_4, random_state=42)
    class_5_resampled = resample(class5, replace=True, n_samples=n_class_5, random_state=42)
    class_6_resampled = resample(class6, replace=True, n_samples=n_class_6, random_state=42)
    class_7_resampled = resample(class7, replace=True, n_samples=n_class_7, random_state=42)
    class_8_resampled = resample(class8, replace=True, n_samples=n_class_8, random_state=42)
    class_9_resampled = resample(class9, replace=True, n_samples=n_class_9, random_state=42)

    #merge datasets back into a single dataframe
    resampled = pd.concat([class_3_resampled, class_4_resampled, class_5_resampled, class_6_resampled, class_7_resampled, class_8_resampled, class_9_resampled])

    #Shuffle the deck
    resampled = resampled.sample(frac=1, random_state=42).reset_index(drop=True)

    return resampled.drop(labels='quality', axis=1), resampled['quality']


def train_model(features_train, target_train):
    # Train model
    model = RandomForestClassifier()
    model.fit(features_train, target_train)
    return model

def model_acc(model, features_test, target_test):
    # Make prediction
    pred = model.predict(features_test)
    # Create evaluation metrics
    f1 = f1_score(target_test, pred, average='weighted')
    acc = accuracy_score(target_test, pred)

    return acc, f1



if __name__ == "__main__":
    df, features_train, target_train, features_val, features_test, target_val, target_test = load_and_split_data()

    print("Baseline Class distribution:", Counter(target_train))

    model = train_model(features_train, target_train)

    #Simulate shifts
    co_shifted_test = sim_covariate_shift(features_test)

    class_rebalance = features_train.join(target_train)

    print(len(class_rebalance), "vs", len(features_train))
    
    label_shifted_features, label_shifted_target = sim_label_shift(features_train.join(target_train))

    print("Current Class distribution:", Counter(label_shifted_target))

    #calculate accuracy considering shifts
    acc, f1 = model_acc(model, features_test, target_test)
    s_acc, s_f1 = model_acc(model, co_shifted_test, target_test)
    #l_acc, l_f1 = model_acc(model, label_shifted_test, target_test)


    print("Accuracy baseline:", acc)
    print("F1 baseline:", f1)
    print("Accuracy with covariate shift:", s_acc)
    print("F1 with covariate shift:", s_f1)
    # print("Accuracy with label shift:", l_acc)
    # print("F1 with label shift:", l_f1)