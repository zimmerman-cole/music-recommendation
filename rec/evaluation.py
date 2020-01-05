"""
Model for evaluating recommendation quality.
"""
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .data_loader import Dataset
from .model import BaseModel


def evaluate_on_valid_set(model, **model_kwargs):
    """
    Args:
    =========================================
    (object) model:
    * A class implementing a method `predict_for_user`, which takes a
      string `user_id` and a np.array with rows of the form: 
      [song_id]
    
    (dict / additional arguments) **model_kwargs:
    * Any keyword arguments to be passed to model.predict().
    =========================================
    """
    msg = "Passed model must implement `predict_for_user()`"
    assert hasattr(model, 'predict_for_user'), msg
    
    valid_data = Dataset(which='valid')
    results = list()
    
    for user_id, true_hidden_songs in tqdm(
        valid_data.iterate_over_hidden_data(), total=len(valid_data)
    ):
        visible_songs = valid_data.fetch_user_data(user_id + '.txt', hidden=False)
        pred_hidden_songs = model.predict_for_user(
            user_id, visible_songs, **model_kwargs
        )
    
        p = compute_precision(pred_hidden_songs, true_hidden_songs[:, 0])
        results.append(p)
        
    mean_avg_precision = np.mean(results)
    return mean_avg_precision


def evaluate_on_test_set(model, **model_kwargs):
    """
    Args:
    =========================================
    (object) model:
    * A class implementing a method `predict_for_user`, which takes a
      string `user_id` and a np.array with rows of the form: 
      [song_id]
    
    (dict / additional arguments) **model_kwargs:
    * Any keyword arguments to be passed to model.predict().
    =========================================
    """
    msg = "Passed model must implement `predict_for_user()`"
    assert hasattr(model, 'predict_for_user'), msg
    
    test_data = Dataset(which='test')
    results = list()
    
    for user_id, true_hidden_songs in tqdm(
        test_data.iterate_over_hidden_data(), total=len(test_data)
    ):
        visible_songs = test_data.fetch_user_data(user_id + '.txt', hidden=False)
        pred_hidden_songs = model.predict_for_user(
            user_id, visible_songs, **model_kwargs
        )
    
        p = compute_precision(pred_hidden_songs, true_hidden_songs[:, 0])
        results.append(p)
        
    mean_avg_precision = np.mean(results)
    return mean_avg_precision


    
# ===========================================================================
# EVALUATION METRICS ========================================================
# ===========================================================================
    
def compute_precision(pred: Iterable, true: Iterable) -> float:
    """
    Returns the precision.
    """
    num_correct = 0
    for t_song in true:
        if t_song in pred:
            num_correct += 1
            
    return num_correct / len(pred)

    
def compute_recall(pred: Iterable, true: Iterable) -> float:
    """
    Returns the recall.
    """
    num_correct = 0
    for t_song in true:
        if t_song in pred:
            num_correct += 1
            
    return num_correct / len(true)
    

def compute_f1_score(pred: Iterable, true: Iterable) -> float:
    num_correct = 0
    for t_song in true:
        if t_song in pred:
            num_correct += 1
    
    precision = num_correct / len(pred)
    recall    = num_correct / len(true)
    
    return (2 * precision * recall) / (precision + recall)

    





























if __name__ == '__main__':
    pass