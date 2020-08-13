from keras.engine.sequential import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.models import load_model
from keras.utils.multi_gpu_utils import multi_gpu_model
def get_model():

    model=load_model("models/CNN_batch64/best_model_64.h5")

    return model
