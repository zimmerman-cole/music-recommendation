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
    # Training, validation, and testing data
    'train_data':         'train_triplets.txt',
    'valid_data_visible': 'evaluation/year1_valid_triplets_visible.txt',
    'valid_data_hidden':  'evaluation/year1_valid_triplets_hidden.txt',
    'test_data_visible':  'evaluation/year1_test_triplets_visible.txt',
    'test_data_hidden':   'evaluation/year1_test_triplets_hidden.txt'
}


class DataLoaderDF(object):
    """
    Data loader which returns data as pandas DataFrames.
    
    
    * In the case of validation or test data, the primary purpose of this class 
      is to load the *visible* data points.
      - Calling __iter__ returns an iterator over the visible points, and
        `load_batch_by_indices` also returns visible data points.

    * Use `load_hidden_data` to load the hidden points for evaluation.
    
    Args:
    ==========================================
    (str) which:
    * Set to one of ['train', 'validation', 'test'] to load the training, 
      validation, or testing data respectively.
    ==========================================
    """
    
    _valid = ['train', 'valid', 'test']
    _columns = ['user_id', 'song_id', 'num_plays']
    
    def __init__(self, which='train'):
        self.which = which
        
        if which == 'train':
            fname = _filenames['train_data']
        elif which == 'valid':
            fname = _filenames['valid_data_visible']
        elif which == 'test':
            fname = _filenames['test_data_visible']
        else:
            raise AssertionError('Please set `which` to any of %s' % DataLoader._valid)
        
        self.data_path = os.path.join(DATA_PATH, fname)
        
        if self.which == 'train':
            self._len = MSDMetadata.num_train_points
        elif self.which == 'valid':
            self._len = MSDMetadata.num_visible_valid_points
        else:
            self._len = MSDMetadata.num_visible_test_points
            
    def __str__(self):
        return 'DataLoaderDF(which=%s, num_data=%d)' % (self.which, len(self))
    
    def __repr__(self):
        return str(self)
    
    def __iter__(self) -> Iterator[pd.Series]:
        """
        Iterates over the *visible* data points deterministically, in the 
        order that they appear in the text file.
        
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

    def load_hidden_data(self, which='valid') -> pd.DataFrame:
        """
        For the [valid/test] data, loads the *ENTIRE* hidden dataset 
        in a pd.DataFrame.
        """
        if self.which == 'valid':
            fname = _filenames['valid_data_hidden']
        elif self.which == 'test':
            fname = _filenames['test_data_hidden']
        elif self.which == 'train':
            raise ValueError('There is no hidden training data.')
        else:
            raise ValueError('Unknown dataset %s' % which)
            
        fpath = os.path.join(DATA_PATH, fname)
        triplets = []
        
        with open(fpath, 'r') as text_file:
            for i, line in enumerate(text_file):
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])

                triplets.append([user_id, song_id, num_plays])

        triplets = pd.DataFrame(
            triplets, columns=['user_id', 'song_id', 'num_plays']
        )

        return triplets
    
    def iterate_over_hidden_user_data(self, which='valid') -> Iterator[tuple]:
        """
        Iterates per-user over the hidden data.
        
        For each user, yields a tuple (user_id, user_data) where
        `user_id` is the ID of the user in question, and
        `user_data` is a list with elements of the form (song_id, num_plays).
        """
        if self.which == 'valid':
            fname = _filenames['valid_data_hidden']
        elif self.which == 'test':
            fname = _filenames['test_data_hidden']
        elif self.which == 'train':
            raise ValueError('There is no hidden training data.')
        else:
            raise ValueError('Unknown dataset %s' % which)
            
        fpath = os.path.join(DATA_PATH, fname)
        
        with open(fpath, 'r') as text_file:
            current_user_id = None
            user_data = []
            
            for i, line in enumerate(text_file):
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])
                
                if (user_id == current_user_id) or (current_user_id is None):
                    user_data.append([song_id, num_plays])
                    
                else:
                    out = (current_user_id, user_data)
                    yield out
                    
                    current_user_id = user_id
                    user_data = [[song_id, num_plays]]

                    
            # yield last user's data
            out = (current_user_id, user_data)
            yield out
        
        


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


# def load_track_info() -> pd.DataFrame:
#     """
#     Loads information 
    
#     Args:
#     ===========================
#     None
#     ===========================
    
#     Returns:
#     ===========================
#     (pd.DataFrame) track_info:
#     * 
#     ===========================
#     """
#     path = os.path.join(DATA_PATH, _filenames['song_to_track'])
    
#     track_info = []
#     with open(path, 'r') as text_file:
#         for line in text_file:
#             line = line.strip().split('<SEP>')
#             song_id, 
            
#             track_info.append([song, tracks])
    
#     track_info = pd.DataFrame(track_info, columns=['song_id', 'track_ids'])
    
#     return track_info





















if __name__ == '__main__':
    pass



