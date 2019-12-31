"""
Module containing the base implementation of a music-recommendation model.
"""

from abc import ABC, abstractmethod
from typing import Iterable


class BaseModel(ABC):
    """
    The base model which all recommender engines should subclass.
    """
    
    @abstractmethod
    def predict_for_user(self, user_data: Iterable) -> Iterable:
        """
        Base method for predicting music recommendations.
        
        All implementations of predict_for_user() should return an iterable with
        elements of the form:  (song_id, num_plays)
        """
        raise NotImplementedError
        
    @abstractmethod
    def fit(self, train_data: Iterable):
        """
        Base method for fitting the recommender to the training data.
        
        The argument `train_data` is (by default) expected to be 
        an instance of data_loader.DataLoaderDF.
        """
        raise NotImplementedError
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass