"""
Module for loading data.
"""
import os

import pandas as pd
from tqdm import tqdm

from .constants import DATA_PATH, MSDMetadata

_filenames = {
    'songs': 'kaggle_songs.txt',
    'users': 'kaggle_users.txt',
    'song_to_track': 'taste_profile_song_to_tracks.txt',
    'evaluation_triplets': 'kaggle_visible_evaluation_triplets.txt',
    'train_triplets': 'train_triplets.txt'
}


def load_song_ids():
    """
    Load the ID associated with each song into a pandas.DataFrame.
    
    Args:
    ===========================
    None
    ===========================
    
    Returns:
    ===========================
    (pd.DataFrame) song_ids:
    * A pandas DataFrame containing the song IDs (in the sole column 'song_id').
    ===========================
    """
    path = os.path.join(DATA_PATH, _filenames['songs'])
    
    song_ids = pd.read_csv(
        path, header=None, names=['song_id', 'id'], sep=' ', index_col=1
    )
    
    return song_ids


def load_user_ids():
    """
    Load the ID associated with each user into a pandas.DataFrame.
    
    Args:
    ===========================
    None
    ===========================
    
    Returns:
    ===========================
    (pd.DataFrame) user_ids:
    * A pandas DataFrame containing the user IDs (in the sole column 'user_id').
    ===========================
    """
    path = os.path.join(DATA_PATH, _filenames['users'])
    
    user_ids = pd.read_csv(path, header=None, names=['user_id'])
    
    return user_ids


def load_song_to_track_data(progress_bar=False):
    """
    Loads the data mapping song IDs to track IDs.
    
    Args:
    ===========================
    (bool) progress_bar:
    * Set progress_bar=True to display a progress bar during data loading.
    ===========================
    
    Returns:
    ===========================
    (pd.DataFrame) song_to_track:
    * A pandas DataFrame containing the mappings from song ID to track ID(s).
    ===========================
    """
    path = os.path.join(DATA_PATH, _filenames['song_to_track'])
    
    song_to_track = []
    with open(path, 'r') as text_file:
        if progress_bar:
            iterator = tqdm(text_file)
        else:
            iterator = text_file
        
        for line in iterator:
            line = line.strip().split('\t')
            song, tracks = line[0], line[1:]
            
            song_to_track.append([song, tracks])
    
    song_to_track = pd.DataFrame(song_to_track, columns=['song_id', 'track_ids'])
    
    return song_to_track


def load_evaluation_triplets(progress_bar=False):
    raise NotImplementedError


def load_train_triplets(progress_bar=False):
    """
    Loads the training data, where each data point is of the form 
    (user_id, song_id, num_plays).
    
    Args:
    ===========================
    (bool) progress_bar:
    * Set progress_bar=True to display a progress bar during data loading.
    ===========================
    
    Returns:
    ===========================
    (pd.DataFrame) train_triplets:
    * A pandas DataFrame containing the train triplets.
    * The columns are ['user_id', 'song_id', 'num_plays'].
    ===========================
    """
    path = os.path.join(DATA_PATH, _filenames['train_triplets'])
    
    train_triplets = []
    with open(path, 'r') as text_file:
        
        if progress_bar:
            iterator = tqdm(text_file, total=MSDMetadata.num_training_points)
        else:
            iterator = text_file
        
        for line in iterator:
            line = line.strip().split('\t')
            user_id, song_id = line[0], line[1]
            num_plays = int(line[2])
            
            train_triplets.append([user_id, song_id, num_plays])
            
    train_triplets = pd.DataFrame(
        train_triplets, columns=['user_id', 'song_id', 'num_plays']
    )
    
    return train_triplets






















if __name__ == '__main__':
    pass



