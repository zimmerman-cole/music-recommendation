"""
Module for loading data.
"""
import os
from typing import Iterable, Iterator

import pandas as pd
import torch
from tqdm import tqdm

from .constants import DATA_PATH, MSDMetadata


# The filenames of each data file
_filenames = {
    'songs': 'kaggle_songs.txt',
    'users': 'kaggle_users.txt',
    'song_to_track': 'taste_profile_song_to_tracks.txt',
    'evaluation_triplets': 'kaggle_visible_evaluation_triplets.txt',
    'train_triplets': 'train_triplets.txt'
}


class DataLoaderDF(object):
    """
    Data loader which returns data as pandas DataFrames.
    """
    
    _valid = ['train', 'evaluation']
    _columns = ['user_id', 'song_id', 'num_plays']
    
    def __init__(self, which='train'):
        self.which = which
        
        if which == 'train':
            fname = _filenames['train_triplets']
        elif which == 'evaluation':
            fname = _filenames['evaluation_triplets']
        else:
            raise AssertionError('Please set `which` to any of %s' % DataLoader._valid)
        
        self.data_path = os.path.join(DATA_PATH, fname)
        
        if self.which == 'train':
            self._len = MSDMetadata.num_training_points
        else:
            self._len = MSDMetadata.num_evaluation_points
            
    def __str__(self):
        return 'DataLoaderDF(which=%s, num_data=%d)' % (self.which, len(self))
    
    def __repr__(self):
        return str(self)
    
    def __iter__(self) -> Iterator[pd.Series]:
        """
        Iterates over the data points deterministically, in the order that they
        appear in the text file.
        
        Yields each data point as a `pandas.Series`.
        """
        with open(self.data_path, 'r') as text_file:
            for line in text_file:
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])
                
                yield pd.Series({
                    'user_id': user_id, 'song_id': song_id, 'num_plays': num_plays
                })
    
    def __len__(self):
        return self._len
    
    def load_batch_by_indices(self, indices: Iterable[int]) -> pd.DataFrame:
        """
        Given an iterable of integer indices, return the corresponding data points
        in a `pandas.DataFrame`.
        
        Does this by iterating over the entire file, yielding each data point if 
        it matches any index passed in `indices`, and returning when either
        all data points in `indices` have been yielded, or the end of file is 
        reached.
        """
        triplets = []
        indices = set(indices)
        
        with open(self.data_path, 'r') as text_file:
            for i, line in enumerate(text_file):
                if i not in indices:
                    continue
                    
                indices.pop(i)
                
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])

                triplets.append([user_id, song_id, num_plays])
                
                if len(indices) == 0:
                    break

        triplets = pd.DataFrame(
            triplets, columns=['user_id', 'song_id', 'num_plays']
        )

        return triplets



def load_song_ids() -> pd.DataFrame:
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


def load_user_ids() -> pd.DataFrame:
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


def load_song_to_track_data(progress_bar=False) -> pd.DataFrame:
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
            iterator = tqdm(text_file, total=MSDMetadata.num_unique_songs)
        else:
            iterator = text_file
        
        for line in iterator:
            line = line.strip().split('\t')
            song, tracks = line[0], line[1:]
            
            song_to_track.append([song, tracks])
    
    song_to_track = pd.DataFrame(song_to_track, columns=['song_id', 'track_ids'])
    
    return song_to_track
























if __name__ == '__main__':
    pass



