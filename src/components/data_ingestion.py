import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import pickle

def automate_data_ingestion(data_path, artifacts_path='artifacts'):
    """
    Automate data ingestion process:
    1. Read raw data from the specified path
    2. Create 'artifacts' directory if it doesn't exist
    3. Process and split the data into training and test sets
    4. Save the processed data and artifacts

    Parameters:
    - data_path (str): Path to the raw data CSV file
    - artifacts_path (str, optional): Directory to store artifacts (default is 'artifacts')
    
    Returns:
    - train_data_path (str): Path to the saved training data
    - test_data_path (str): Path to the saved test data
    """
    
    # Create 'artifacts' directory if it doesn't exist
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    # Read the data
    df = pd.read_csv(data_path)
    df = df.iloc[:100001]

    # Add an ID column
    df['ID'] = df.reset_index().index + 1

    # Drop specified columns
    columns_to_drop = ['duration_ms', 'energy', 'liveness', 'popularity',
                       'explicit', 'speechiness', 'acousticness', 'instrumentalness', 
                       'loudness', 'tempo', 'danceability', 'mode','valence','id']
    df.drop(columns=columns_to_drop, inplace=True)

    ordinal_columns = ['year', 'key']
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    ordinal_transformer = OrdinalEncoder()
    scaler = StandardScaler()

    numerical_transformer = Pipeline(steps=[
        ('scaler', scaler)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', ordinal_transformer, ordinal_columns)
        ])

    X = df.drop('name', axis=1)
    y = df['name']
    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save raw data as CSV
    df.to_csv(os.path.join(artifacts_path, 'raw_data.csv'), index=False)

    # Save train and test data as CSV
    train_data = pd.DataFrame(X_train, columns=numerical_columns + ordinal_columns)
    train_data['name'] = y_train.values
    train_data_path = os.path.join(artifacts_path, 'train_data.csv')
    train_data.to_csv(train_data_path, index=False)

    test_data = pd.DataFrame(X_test, columns=numerical_columns + ordinal_columns)
    test_data['name'] = y_test.values
    test_data_path = os.path.join(artifacts_path, 'test_data.csv')
    test_data.to_csv(test_data_path, index=False)

    # Save preprocessor
    preprocessor_path = os.path.join(artifacts_path, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    return train_data_path, test_data_path

if __name__ == "__main__":
    data_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\data\music.csv'
    train_data_path, test_data_path = automate_data_ingestion(data_path)
    print(f"Training data saved at: {train_data_path}")
    print(f"Test data saved at: {test_data_path}")