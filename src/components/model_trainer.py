import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_function

class MusicRecommendationModelConfig:
    trained_model_file_path = os.path.join("artifacts", "music_recommendation_model.pkl")

class MusicRecommendationModel:
    def __init__(self):
        self.config = MusicRecommendationModelConfig()
        self.preprocessor = None
        self.model = None
        self.X_train = None
        self.df = None  # Instance variable for the DataFrame

    def preprocess_data(self, df):
        ordinal_columns = ['year', 'key', 'mode']
        numerical_columns = ['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo']
        
        for col in numerical_columns:
            df[col].fillna(df[col].median(), inplace=True)
        
        ordinal_transformer = OrdinalEncoder()
        scaler = StandardScaler()
        numerical_transformer = Pipeline(steps=[('scaler', scaler)])
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', ordinal_transformer, ordinal_columns)
        ])
        
        X = df.drop('name', axis=1)
        y = df['name']
        X = self.preprocessor.fit_transform(X)
        self.X_train, _, _, self.df = train_test_split(X, y, df, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.model.fit(self.X_train)

    def find_song_index(self, song_name, data):
        try:
            index = data[data['name'] == song_name].index[0]
            return index
        except IndexError:
            logging.error(f"Song '{song_name}' not found in the dataset.")
            raise CustomException(f"Song '{song_name}' not found.", sys)

    def recommend_songs(self, song_name):
        song_index = self.find_song_index(song_name, self.df)  # Access df from instance variable
        if song_index is None:
            return None
        
        # Get the feature vector of the song
        song_vector = self.X_train[song_index]
        
        # Reshape the vector to be compatible with the model's input shape
        song_vector = song_vector.reshape(1, -1)
        
        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(song_vector)
        
        # Get the song names of the neighbors
        recommended_songs = self.df.iloc[indices[0]]['name'].values
        
        return recommended_songs

if __name__ == "__main__":
    try:
        music_model = MusicRecommendationModel()
        
        # Preprocess the data
        df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\data\music.csv')
        df = df.iloc[:100001]
        music_model.preprocess_data(df)

        # Train the model
        music_model.train_model()

        song_name = 'Danny Boy'
        recommended_songs = music_model.recommend_songs(song_name)
        
        if recommended_songs is not None:
            print(f"Recommended songs for {song_name}: {recommended_songs}")

    except CustomException as ce:
        print(f"Custom Exception occurred: {ce}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")