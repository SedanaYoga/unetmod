import numpy as np 
import os 
import skimage.io as io 
import skimage.transform as trans 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None, input_size = (256,256, 1)):
    inputs = Input(input_size)
    #keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
    #                   padding='valid', data_format=None, 
    #                   dilation_rate=(1, 1), activation=None, 
    #                   use_bias=True, kernel_initializer='glorot_uniform', 
    #                   bias_initializer='zeros', kernel_regularizer=None, 
    #                   bias_regularizer=None, activity_regularizer=None, 
    #                   kernel_constraint=None, bias_constraint=None)
    
    # Rangkaian Extraction 1
    conv1 = Conv2D(64, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

    # Rangkaian Extraction 2
    conv2 = Conv2D(128, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size = (2,2))(conv2)

    # Rangkaian Extraction 3
    conv3 = Conv2D(256, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

    # Rangkaian Extraction 4
    conv4 = Conv2D(512, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(drop4)

    # Rangkaian Extraction 5
    conv5 = Conv2D(1024, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    drop5 = Dropout(0.5)(conv5)

    # Rangkaian Expansion 1
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    # Rangkaian Expansion 2
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    # Rangkaian Expansion 3
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    # Rangkaian Expansion 4
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    
    # Rangkaian Expansion 5
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Membuat Model 
    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model