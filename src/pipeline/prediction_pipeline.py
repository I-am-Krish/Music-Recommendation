import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import load_function

class MusicRecommendationModelConfig:
    preprocessor_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\artifacts\preprocessor.pkl'
    model_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\artifacts\model.pkl'

class MusicRecommendationModel:
    def __init__(self):
        self.config = MusicRecommendationModelConfig()
        self.preprocessor = None
        self.model = None
        self.df = None  # Instance variable for the DataFrame

    def load_pretrained_model(self):
        self.preprocessor = load_function(file_path=self.config.preprocessor_path)
        self.model = load_function(file_path=self.config.model_path)
        
        # Assuming you have a path to your CSV data
        data_path = r'C:\Users\krish\OneDrive\Desktop\Music Recomendation\notebooks\data\music.csv'
        df = pd.read_csv(data_path)
        df = df.iloc[:100001]
        
        # Add an 'ID' column with sequential numbering starting from 1
        df['ID'] = df.reset_index().index + 1
        
        self.df = df  # Initialize self.df with the loaded DataFrame

    def find_song_index(self, song_name, data):
        try:
            index = data[data['name'] == song_name].index[0]
            return index
        except IndexError:
            logging.error(f"Song '{song_name}' not found in the dataset.")
            raise CustomException(f"Song '{song_name}' not found.")

    def recommend_songs(self, song_name):
        if self.df is None:
            raise CustomException("DataFrame is not initialized.")
        
        # Reset index to create the 'ID' column
        self.df.reset_index(drop=True, inplace=True)
        
        song_features = self.df[self.df['name'] == song_name].drop('name', axis=1)
        song_features_preprocessed = self.preprocessor.transform(song_features)
        
        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(song_features_preprocessed)
        
        # Get the song names of the neighbors
        recommended_songs = self.df.iloc[indices[0]]['name'].values
        
        return recommended_songs