import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data():
    # Load dataset
    data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', header=0, index_col='PassengerId')
    return data


def show_nulls(data):
    # Identify columns with missing data
    print("Missing values: " + str(data.isnull().sum()))

def replace_nulls(data):
    # Impute missing values
    no_null_data = pd.DataFrame()
    for column in data.columns:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            no_null_data[column] = data[column].fillna(data[column].median())
        else:
            no_null_data[column] = data[column]
        
    return no_null_data


def scale_features(features):

    # Select only the numeric columns for scaling
    numeric_columns = ['Age', 'Fare']

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the numeric columns
    features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
    
    return features



if __name__ == '__main__':
    #Get titanic dataset to start
    titanic_data = load_data()

    # Separate features and target variable
    titanic_features = titanic_data.drop(columns=['Survived'])
    titanic_target = titanic_data['Survived']

    #See nulls in the dataset
    show_nulls(titanic_features)
    #Replace nulls
    no_null_titanic = replace_nulls(titanic_features)
    # Verify no missing values
    show_nulls(no_null_titanic)

    # Pre-scale of features
    print(no_null_titanic.describe())
    scaled_titanic = scale_features(no_null_titanic)
    #Verify Scaling
    print(scaled_titanic.describe())

