import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


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

def sim_label_shift(test_df, target_distribution):
    # Determine the number of samples needed for each label to match the target distribution
    total_samples = len(test_df)
    samples_needed = {label: int(total_samples * target_distribution[label]) for label in target_distribution}
    
    # Create a new test set with the desired distribution
    new_test_df = pd.DataFrame()
    for label, count in samples_needed.items():
        label_samples = test_df[test_df['quality'] == label].sample(count, replace=True, random_state=42)
        new_test_df = pd.concat([new_test_df, label_samples])
    
    return new_test_df

def sim_concept_shift(test_data):
    test_data['alcohol'] += np.random.normal(2, 0.5, size=test_data.shape[0])
    return test_data

def train_model(features_train, target_train):
    # Train model
    model = RandomForestClassifier()
    model.fit(features_train, target_train)
    return model

def model_acc(pred, target_test):
    # Create evaluation metrics
    f1 = f1_score(target_test, pred, average='weighted')
    acc = accuracy_score(target_test, pred)
    return acc, f1

def make_prediction(model, features):
    pred = model.predict(features) 
    return pred

def confu_matrix(pred, true_set, name):
    cm = confusion_matrix(true_set, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['3', '4', '5', '6', '7', '8'], yticklabels=['3', '4', '5', '6', '7', '8'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(name)
    plt.show()



if __name__ == "__main__":
    #load data and train model
    df, features_train, target_train, features_val, features_test, target_val, target_test = load_and_split_data()
    model = train_model(features_train, target_train)

    #Identify baseline
    print("Baseline Class distribution:", Counter(target_test))
    base_pred = make_prediction(model, features_test)
    acc, f1 = model_acc(base_pred, target_test)
    print("Accuracy baseline:", acc)
    print("F1 baseline:", f1)
    confu_matrix(base_pred, target_test, 'Baseline')

    #Simulate covariate shift
    covar_shifted_test = sim_covariate_shift(features_test)
    covar_pred = make_prediction(model, covar_shifted_test)
    s_acc, s_f1 = model_acc(covar_pred, target_test)
    print("Accuracy with covariate shift:", s_acc)
    print("F1 with covariate shift:", s_f1)
    confu_matrix(covar_pred, target_test, 'Covariate Shift')

    #simulate label shift and confirm class distribution change.
    whole_test_set = features_test.join(target_test)
    target_distribution = {3: .1, 4: .1, 5:.2, 6:.3, 7:.2, 8:.1}
    label_shifted_set = sim_label_shift(whole_test_set, target_distribution)
    print("Class distribution after label shift:", Counter(label_shifted_set['quality']))

    #Generate performance metrics considering label shift
    label_pred = make_prediction(model, label_shifted_set.drop('quality', axis=1))
    l_acc, l_f1 = model_acc(label_pred, label_shifted_set['quality'])
    print("Accuracy with label shift:", l_acc)
    print("F1 with label shift:", l_f1)
    confu_matrix(label_pred, label_shifted_set['quality'], 'Label Shift')

    #Simulate concept shift
    con_shifted_test = sim_concept_shift(features_test)
    con_pred = make_prediction(model, con_shifted_test)
    s_acc, s_f1 = model_acc(con_pred, target_test)
    print("Accuracy with Concept shift:", s_acc)
    print("F1 with Concept shift:", s_f1)
    confu_matrix(con_pred, target_test, 'Concept Shift')