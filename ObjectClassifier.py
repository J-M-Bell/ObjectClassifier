import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
import re
import io
import keras
import random
from nltk.tokenize import word_tokenize
from keras import Input, activations
from keras.callbacks import ModelCheckpoint
from keras.layers import SimpleRNN, Dense, LSTM, Dropout, Embedding
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.models import Sequential
from keras.datasets import cifar10
from sklearn.preprocessing import OrdinalEncoder


data= cifar10.load_data()
