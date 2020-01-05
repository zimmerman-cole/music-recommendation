"""
Module for managing Million Song Database (MSD) data.
"""
import os
from typing import Iterable, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from rec.constants import DATA_PATH

# ===================================================================
# Attributes+methods hidden from other modules ====================== 
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


def _get_song_index(song_id: str) -> int:
    """
    Given a MSD-format song ID, return the corresponding integer index
    from `msdutils.song_ids`.
    
    Can be used to create a vector of the songs each user has and has not
    listened to  -- see `create_user_vector()`.
    """
    return song_ids[song_ids['song_id'] == song_id].index[0]


# ===================================================================
# Visible attributes + methods
# ===================================================================

song_ids = _load_song_ids()
user_ids = _load_user_ids()
song_to_track = _load_song_to_track_data()


def create_user_vector(
    user_data: Iterable, include_play_counts=False
) -> sps.coo_matrix:
    """
    Given a list of the [(song_id, num_plays)] data for a specific user,
    returns a vector containing the play count information for each song for the user.
    
    Depending on whether `include_play_counts=True` or not, the entries will 
    be binary (False) or non-negative integers (True).
    
    ============================================
    Args:
    ============================================
    (Iterable) user_data:
    * An iterable of the [(song_id, num_plays)] data for the given user.
    
    (bool) include_play_counts:
    * Whether to return a binary vector (has the user played/not played each song?)
      or a non-negative integer vector (where each entry contains the number of 
      times the user played each song). 
    * Note that the majority of the entries will be zero.
    ============================================
    
    Returns:
    ============================================
    (scipy.sparse.coo_matrix) user_vector
    * Returns the vector as a matrix of shape (1, num_unique_songs).
    ============================================
    """
    song_idxs = []
    song_data = []
    for song_id, num_plays in user_data:
        song_idxs.append(_get_song_index(song_id))
        song_data.append(num_plays if include_play_counts else 1)
    
    num_unique_songs = song_ids.values.shape[0]
    return sps.coo_matrix(
        (song_data, ([0]*len(song_idxs), song_idxs)), 
        shape=(1, num_unique_songs),
        dtype=int
    )


def create_occurrence_matrix(
    dataset, included_users: Union[str, Iterable[str], Iterable[int]] = 'all',
    include_play_counts=False
) -> sps.coo_matrix:
    """
    Given a dataset of (user_id, song_id, num_plays) triplets, creates a 
    occurrence matrix to be used downstream in a latent semantic analysis-based
    recommender system (or otherwise).
    
    ============================================
    Args:
    ============================================
    (rec.data_loader.Dataset) dataset:
    * A dataset implementing the method `iterate_over_visible_data()`.
    
    (Union[str, Iterable[str], Iterable[int]]) included_users:
    * This argument controls which users' data is included in the matrix. This
      is done because the number of data is too high to fit in memory, even
      as a sparse matrix.
    * It has three valid values:
        (1) 'all':
            - Includes all users. Will require very large amounts of memory.
        (2) Iterable[str] of user IDs: 
            - Each user is included if their corresponding user ID is included
              in this list/set/...
        (3) Iterable[int] of integer indices:        
            - Each user is included if their position in which they occur in 
              the full (text file) dataset is included in this list/set/...
            - This can be used if the exact group of users included in this
              matrix does not matter.
    
    (bool) include_play_counts:
    * Whether to return a binary vector (has the user played/not played each song?)
      or a non-negative integer vector (where each entry contains the number of 
      times the user played each song). 
    * Note that the majority of the entries will be zero.
    ============================================
    
    Returns:
    ============================================
    (scipy.sparse.coo_matrix) occurrence_matrix
    * Returns a matrix with shape (num_users, num_unique_songs).
    ============================================
    """
    if isinstance(included_users, str) and included_users == 'all':
        include = lambda ids: True
        
    # assume included_users is a list of user IDs
    elif isinstance(included_users[0], str):
        include = lambda ids: ids[0] in included_users
        
    # assume included_users is a list of integer indices of users
    elif isinstance(included_users[0], np.integer):
        include = lambda ids: ids[1] in included_users
        
    else:
        raise ValueError("Invalid argument 'included_users' (got %s)" % included_users)
    
    
    rows = []
    cols = []
    data = []
    ctr = 0
    for int_id, (user_id, user_data) in enumerate(dataset.iterate_over_visible_data()):
        if not include((user_id, int_id)):
            continue
            
        for (song_id, num_plays) in user_data:
            rows.append(ctr)
            cols.append(_get_song_index(song_id))
            data.append(num_plays if include_play_counts else 1)
            
        ctr += 1
        
    num_rows = len(dataset) if included_users == 'all' else len(included_users)
    num_unique_songs = song_ids.values.shape[0]
        
    return sps.coo_matrix(
        (data, (rows, cols)), shape=(num_rows, num_unique_songs), dtype=int
    )
        
        
    
    
    
    
    







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass

