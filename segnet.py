import numpy as np 
import os 
import skimage.io as io 
import skimage.transform as trans 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def segnet(pretrained_weights = None, input_size = (256,256, 1)):
    img_input = Input(shape=input_size)
    x = img_input
    #Encoder
    x = Conv2D(64, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    #50x50
    x = Conv2D(256, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    #25x25
    x = Conv2D(512, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    #Decoder
    x = Conv2D(512, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #25x25
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(256, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #50x50
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(128, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #100x100
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(1, 1, padding = "valid")(x)

    x = Reshape((256,256,1))(x)
    x = Activation("softmax")(x)

    model = Model(inputs = img_input, outputs = x)

    model.compile(optimizer = rmsprop(lr = 1e-4, decay = 1e-6) , loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model