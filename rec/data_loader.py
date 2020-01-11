"""
Module for loading data.
"""
import os
from typing import Iterator, Union

import numpy as np
import pandas as pd
import torch
from dask import dataframe as ddf

from .constants import DATA_PATH, MSDMetadata


class _DataPathFetcher(object):
    """
    TODO: clean this up -- or better yet, use mySQL backend to manage the data
    """
    
    _valid = ['train', 'valid', 'test']
    
    _data_paths = {
        'train_data'        : 'triplets/train/',
        'valid_data_visible': 'triplets/valid/visible/',
        'valid_data_hidden' : 'triplets/valid/hidden/',
        'test_data_visible' : 'triplets/test/visible/',
        'test_data_hidden'  : 'triplets/test/hidden/'
    }
    
    _user_lists = {
        'train': 'triplets/train_users.txt',
        'valid': 'triplets/valid_users.txt',
        'test' : 'triplets/test_users.txt'
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
        the corresponding 7-tuple:
          (
            path_to_visible_triplet_directory,
            path_to_hidden_triplet_directory,    # if which != 'train', else None
            path_to_full_visible_triplet_file,
            path_to_full_hidden_triplet_file,    # if which != 'train', else None
            path_to_user_list, 
            num_visible_triplets,                # num visible data points
            num_unique_users
          )
        """
        if which == 'train':
            fname             = cls._data_paths['train_data']
            fname_hidden      = None
            fname_full        = cls._full_files['train_data']
            fname_full_hidden = None
            user_fname        = cls._user_lists['train']
            num_points        = MSDMetadata.num_train_points
            num_users         = MSDMetadata.num_train_users
            
        elif which == 'valid':
            fname             = cls._data_paths['valid_data_visible']
            fname_hidden      = cls._data_paths['valid_data_hidden']
            fname_full        = cls._full_files['valid_data_visible']
            fname_full_hidden = cls._full_files['valid_data_hidden']
            user_fname        = cls._user_lists['valid']
            num_points        = MSDMetadata.num_visible_valid_points
            num_users         = MSDMetadata.num_valid_users
            
        elif which == 'test':
            fname             = cls._data_paths['test_data_visible']
            fname_hidden      = cls._data_paths['test_data_hidden']
            fname_full        = cls._full_files['test_data_visible']
            fname_full_hidden = cls._full_files['test_data_hidden']
            user_fname        = cls._user_lists['test']
            num_points        = MSDMetadata.num_visible_test_points
            num_users         = MSDMetadata.num_test_users
            
        else:
            raise AssertionError(
                'Please set `which` to any of %s' % DataLoader._valid
            )
        
        data_path      = os.path.join(DATA_PATH, fname)
        data_path_full = os.path.join(DATA_PATH, fname_full)
        user_path      = os.path.join(DATA_PATH, user_fname)
        
        if which == 'train':
            data_path_hidden      = None
            data_path_hidden_full = None
        else:
            data_path_hidden      = os.path.join(DATA_PATH, fname_hidden)
            data_path_hidden_full = os.path.join(DATA_PATH, fname_full_hidden)
            
        return (
            data_path, 
            data_path_hidden, 
            data_path_full, 
            data_path_hidden_full,
            user_path,
            num_points, 
            num_users
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
        self.user_list_path        = data_paths[4]
        self.num_points            = data_paths[5]
        self.num_users             = data_paths[6]
        
        self.user_list = []
        with open(self.user_list_path, 'r') as text_file:
            self.user_list = [line.strip() for line in text_file]
    
    def __len__(self):
        """
        Returns the number of unique users.
        """
        return self.num_users
    
    def __getitem__(self, key: Union[int, str]) -> np.array:
        """
        Retrieve a user's data given one of:
        (1) their ID's position in self.user_list   (integer)
        (2) their user ID                           (string)
         
        This functionality is primarily for compatibility with 
        standard PyTorch samplers.
        """
        if isinstance(key, np.integer):
            key = self.user_list[key]
        
        return self.fetch_user_data(key)
    
    def fetch_user_data(self, user_id: str, hidden=False) -> np.array:
        if hidden:
            path = self.data_path_hidden
        else:
            path = self.data_path
        
        file_path = os.path.join(path, user_id + '.txt')
        with open(file_path, 'r') as text_file:
            user_data = []
            
            for line in text_file:
                line = line.strip().split(', ')
                song_id = line[0]
                num_plays = int(line[1])
                user_data.append([song_id, num_plays])
                
        return np.array(user_data, dtype=object).reshape(-1, 2)
                
    def iterate_over_visible_data(self) -> Iterator[tuple]:
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
        if self.which == 'train':
            raise ValueError('There is no hidden training data.')
        
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

    def load_dask_dataframe(self, hidden=False) -> ddf.DataFrame:
        """
        Returns a dask dataframe containing the whole dataset.
        """
        if hidden:
            path = self.data_path_hidden_full
        else:
            path = self.data_path_full
            
        data_df = ddf.read_csv(
            path, sep='\t', header=None, 
            names=['user_id', 'song_id', 'num_plays'],
            dtype={'user_id': 'category', 'song_id': 'category', 'num_plays': int}
        )
        
        return data_df.categorize(columns=['song_id'])






















if __name__ == '__main__':
    pass



