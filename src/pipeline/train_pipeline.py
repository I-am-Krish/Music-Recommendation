import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.utils import load_function
from src.exception import CustomException

# Load the data
data_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\data\music.csv'
df = pd.read_csv(data_path)
df = df.iloc[:100001]

# Absolute paths for preprocessor and model
preprocessor_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\artifacts\preprocessor.pkl'
model_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\artifacts\model.pkl'

try:
    preprocessor = load_function(file_path=preprocessor_path)
    model = load_function(file_path=model_path)
except Exception as e:
    print(f"Error loading files: {e}")

def preprocess_input(input_data, preprocessor):
    # Preprocess the input data
    X_preprocessed = preprocessor.transform(input_data)
    
    return X_preprocessed

def recommend_songs(input_data, model, preprocessor):
    # Preprocess the input song data
    X_input = preprocess_input(input_data, preprocessor)
    
    # Find the nearest neighbors
    distances, indices = model.kneighbors(X_input)
    
    # Get the recommended song names
    recommended_songs = df.iloc[indices[0]]['name'].values
    
    return recommended_songs

if __name__ == "__main__":
    try:
        input_data = {
            'year': [1921],
            'artists': ["['Frank Parker']"],
            'key': [5],
            'name': ['Danny Boy'],
            'ID': [3]
        }

        input_df = pd.DataFrame(input_data)
        
        recommended_songs = recommend_songs(input_df, model, preprocessor)
        
        if recommended_songs is not None:
            print(f"Recommended songs for {input_df['name'].values[0]}: {recommended_songs}")

    except CustomException as ce:
        print(f"Custom Exception occurred: {ce}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")