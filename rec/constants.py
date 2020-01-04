"""
Stores constant values for use in the rest of the program, 
e.g. the path to the root project directory.
"""
import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, '../'))
DATA_PATH = os.path.join(ROOT_PATH, 'data/')


class MSDMetadata(object):
    """
    Metadata for the Million Songs Dataset.
    
    - See http://millionsongdataset.com/ for details.
    - Data was sourced from Kaggle @ https://www.kaggle.com/c/msdchallenge/data.
    """
    columns = ['user_id', 'song_id', 'num_plays']
    
    # Number of unique users, songs, and tracks
    num_unique_users = 110000  # (in the validation+test sets)
    num_unique_songs = 386213
    num_unique_tracks = 1000000
    
    # total number of training triplets
    # (full listening histories of the users)
    num_train_points = 48373586
    num_train_users = 1019317
    
    # total number of *visible* / *hidden* validation triplets
    # from data/evaluation/year1_valid_triplets_[hidden/visible].txt
    num_visible_valid_points = 131039
    num_hidden_valid_points  = 135938
    num_valid_users = 10000
    
    # total number of *visible* / *hidden* test triplets
    # from data/evaluation/year1_test_triplets_[hidden/visible].txt
    num_visible_test_points = 1319894
    num_hidden_test_points  = 1368430
    num_test_users = 99999
    
    