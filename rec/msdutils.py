"""
Module for managing Million Song Database (MSD) data.
"""
import os

import pandas as pd

from rec.data_loader import load_song_ids

# ===================================================================
# ===================================================================
# The filenames of each data file
_filenames = {
    'songs':         'msd/kaggle_songs.txt',
    'users':         'msd/kaggle_users.txt',
    'song_to_track': 'msd/taste_profile_song_to_tracks.txt',
}


def _load_song_ids() -> pd.DataFrame:
    """
    Load the mapping from song ID associated with each song into a pandas.DataFrame.
    
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


def _load_user_ids() -> pd.DataFrame:
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


def _load_song_to_track_data() -> pd.DataFrame:
    """
    Loads the data mapping song IDs to track IDs.
    
    Args:
    ===========================
    None
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
        for line in text_file:
            line = line.strip().split('\t')
            song, tracks = line[0], line[1:]
            
            song_to_track.append([song, tracks])
    
    song_to_track = pd.DataFrame(song_to_track, columns=['song_id', 'track_ids'])
    
    return song_to_track

# ===================================================================
# ===================================================================

song_ids = _load_song_ids()
user_ids = _load_user_ids()
song_to_track = _load_song_to_track_data()








if __name__ == '__main__':
    pass

