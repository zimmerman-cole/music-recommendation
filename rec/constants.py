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
    num_training_points = 48373586
    num_unique_users = 110000
    num_unique_songs = 386213
    
    num_evaluation_points = 1450933