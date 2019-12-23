"""
Stores constant values for use in the rest of the program, 
e.g. the path to the root project directory.
"""
import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data/')
