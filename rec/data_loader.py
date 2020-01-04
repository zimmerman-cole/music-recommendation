"""
Module for loading data.
"""
import os
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch

from .constants import DATA_PATH, MSDMetadata


# The filenames of each data file
_filenames = {
    'songs':         'msd/kaggle_songs.txt',
    'users':         'msd/kaggle_users.txt',
    'song_to_track': 'msd/taste_profile_song_to_tracks.txt',
}


class _DataPathFetcher(object):
    
    _valid = ['train', 'valid', 'test']
    
    _data_paths = {
        'train_data'        : 'triplets/train/',
        'valid_data_visible': 'triplets/valid/visible/',
        'valid_data_hidden' : 'triplets/valid/hidden/',
        'test_data_visible' : 'triplets/test/visible/',
        'test_data_hidden'  : 'triplets/test/hidden/'
    }
    
    _full_files = {
        'train_data'        : 'triplets/full_files/train_triplets.txt',
        'valid_data_visible': 'triplets/full_files/year1_valid_triplets_visible.txt',
        'valid_data_hidden' : 'triplets/full_files/year1_valid_triplets_hidden.txt',
        'test_data_visible' : 'triplets/full_files/year1_test_triplets_visible.txt',
        'test_data_hidden'  : 'triplets/full_files/year1_test_triplets_hidden.txt'
    }

    @classmethod
    def fetch(cls, which):
        """
        Given a dataset choice \in ['train', 'valid', 'test'], returns
        the corresponding 6-tuple:
          (
            path_to_visible_triplet_directory,
            path_to_hidden_triplet_directory,    # if which != 'train', else None
            path_to_full_visible_triplet_file,
            path_to_full_hidden_triplet_file,    # if which != 'train', else None
            num_visible_triplets,                # num visible data points
            num_unique_users
          )
        """
        if which == 'train':
            fname             = cls._data_paths['train_data']
            fname_hidden      = None
            fname_full        = cls._full_files['train_data']
            fname_full_hidden = None
            num_points        = MSDMetadata.num_train_points
            num_users         = MSDMetadata.num_train_users
            
        elif which == 'valid':
            fname             = cls._data_paths['valid_data_visible']
            fname_hidden      = cls._data_paths['valid_data_hidden']
            fname_full        = cls._full_files['valid_data_visible']
            fname_full_hidden = cls._full_files['valid_data_hidden']
            num_points        = MSDMetadata.num_visible_valid_points
            num_users         = MSDMetadata.num_valid_users
            
        elif which == 'test':
            fname             = cls._data_paths['test_data_visible']
            fname_hidden      = cls._data_paths['test_data_hidden']
            fname_full        = cls._full_files['test_data_visible']
            fname_full_hidden = cls._full_files['test_data_hidden']
            num_points        = MSDMetadata.num_visible_test_points
            num_users         = MSDMetadata.num_test_users
            
        else:
            raise AssertionError(
                'Please set `which` to any of %s' % DataLoader._valid
            )
        
        data_path      = os.path.join(DATA_PATH, fname)
        data_path_full = os.path.join(DATA_PATH, fname_full)
        
        if which == 'train':
            data_path_hidden      = None
            data_path_hidden_full = None
        else:
            data_path_hidden      = os.path.join(DATA_PATH, fname_hidden)
            data_path_hidden_full = os.path.join(DATA_PATH, fname_full_hidden)
            
        return (
            data_path, data_path_hidden, 
            data_path_full, data_path_hidden_full,
            num_points, num_users
        )


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch-style data set of the hidden/visible train/valid/test data.
    
    Currently loads data on a *per-user* basis, i.e. its main functionality
    is to load a list of [(song_id, num_plays)] data points for each user.
    """
    
    _ignore = {'.ipynb_checkpoints'}
    
    def __init__(self, which='train'):
        super(Dataset, self).__init__()
        self.which  = which
        
        data_paths = _DataPathFetcher.fetch(which=which)
        self.data_path             = data_paths[0]
        self.data_path_hidden      = data_paths[1]
        self.data_path_full        = data_paths[2]
        self.data_path_hidden_full = data_paths[3]
        self.num_points            = data_paths[4]
        self.num_users             = data_paths[5]
    
    def __len__(self):
        """
        Returns the number of unique users.
        """
        return self.num_users
    
    def __getitem__(self, user_id: str) -> np.array:
        return self.fetch_user_data(user_id + '.txt')
    
    def fetch_user_data(self, fname: str, hidden=False) -> np.array:
        if hidden:
            path = self.data_path_hidden
        else:
            path = self.data_path
        
        file_path = os.path.join(path, fname)
        with open(file_path, 'r') as text_file:
            user_data = []
            
            for line in text_file:
                line = line.strip().split(', ')
                song_id = line[0]
                num_plays = int(line[1])
                user_data.append([song_id, num_plays])
                
        return np.array(user_data, dtype=object).reshape(-1, 2)
                
    def iterate_over_visible_data(self) -> Iterable[tuple]:
        """
        Iterates per-user over the visible data in a deterministic order.
        
        For each user, yields a tuple (user_id, user_data) where
        `user_id` is the ID of the user in question, and
        `user_data` is a np.array with rows of the form [song_id, num_plays].
        """
        with open(self.data_path_full, 'r') as text_file:
            first_line = text_file.readline().strip().split('\t')
            user_id, song_id = first_line[0], first_line[1]
            num_plays = int(first_line[2])
            
            current_user_id = user_id
            current_user_data = [(song_id, num_plays)]
            
            for line in text_file:
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])

                if user_id != current_user_id:
                    yield (current_user_id, np.array(current_user_data).reshape(-1, 2))

                    # reset data for new user
                    current_user_id = user_id
                    current_user_data = [(song_id, num_plays)]
                else:
                    current_user_data.append((song_id, num_plays))
        
        # yield the last user's data
        yield (current_user_id, np.array(current_user_data).reshape(-1, 2))
            
    def iterate_over_hidden_data(self) -> Iterator[tuple]:
        """
        Iterates per-user over the hidden data in a deterministic order.
        
        For each user, yields a tuple (user_id, user_data) where
        `user_id` is the ID of the user in question, and
        `user_data` is a np.array with rows of the form [song_id, num_plays].
        """
        with open(self.data_path_hidden_full, 'r') as text_file:
            first_line = text_file.readline().strip().split('\t')
            user_id, song_id = first_line[0], first_line[1]
            num_plays = int(first_line[2])
            
            current_user_id = user_id
            current_user_data = [(song_id, num_plays)]
            
            for line in text_file:
                line = line.strip().split('\t')
                user_id, song_id = line[0], line[1]
                num_plays = int(line[2])

                if user_id != current_user_id:
                    yield (current_user_id, np.array(current_user_data).reshape(-1, 2))

                    # reset data for new user
                    current_user_id = user_id
                    current_user_data = [(song_id, num_plays)]
                else:
                    current_user_data.append((song_id, num_plays))
        
        # yield the last user's data
        yield (current_user_id, np.array(current_user_data).reshape(-1, 2))


def load_song_ids() -> pd.DataFrame:
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


def load_song_to_track_data() -> pd.DataFrame:
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



