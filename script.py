

from cross_validation import *
from tmp_util import *
from models import *
from word_adjacency_matrix import *


def init():
    train_x,train_y,test_x = load_data()
    
    cross_validation(train_x,train_y,10,Ensemble_driver)
    
