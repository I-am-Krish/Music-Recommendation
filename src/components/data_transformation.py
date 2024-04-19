import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import pickle
from sklearn.model_selection import train_test_split

try:
    # Read the data
    df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\data\music.csv')

    # Drop specified columns
    columns_to_drop = ['duration_ms', 'energy', 'liveness', 'popularity',
                       'explicit', 'speechiness', 'acousticness', 'instrumentalness', 
                       'loudness', 'tempo', 'danceability', 'mode','valence','id']
    df.drop(columns=columns_to_drop, inplace=True)

    # Handle missing values in numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Define transformers
    ordinal_columns = ['year', 'key']
    ordinal_transformer = OrdinalEncoder()
    scaler = StandardScaler()

    # Create numerical pipeline
    numerical_transformer = Pipeline(steps=[
        ('scaler', scaler)
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', ordinal_transformer, ordinal_columns)
        ])

    # Separate features and target
    X = df.drop('name', axis=1)
    y = df['name']

    # Apply the preprocessor to the features
    X_transformed = preprocessor.fit_transform(X)

    # Split the transformed data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Save the preprocessor
    with open('artifacts/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("Data transformation and preprocessor saving completed successfully!")

except Exception as e:
    print(f"An error occurred: {e}")