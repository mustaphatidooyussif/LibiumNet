"""
Lip reading model: A model that predicts the words of a spoken mouth 
in a silence video. 

@authors: Mustapha Tidoo Yussif, Samuel Atule, Jean Sabastien Dovonon, 
          and Nutifafa Amedior. 
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, TimeDistributed, LSTM, Input

tf.enable_eager_execution()

class LibiumNet(object):
    """TA lipreading model, `LibiunNet`
    This is lip reading model which reads or predicts the words of a spoken mouth in a silent video. 
    This model implements the RCNN (Recurrent Convulutional Neural Network) architecture. 

    :param img_c: The number of channels of the input image. i.e. a frame in a video (default 3).
    :param img_w: The width of the input image i.e. a frame in a video (default 256)
    :param img_h: The height of the input image i.e. a frame in a video (default 256)
    :param frames_n: The total number of frames in an input video (default 29)
    :param output_size: The output size of the network. 
    
    """
    def __init__(self, img_c=3, img_w=256, img_h=256, frames_n=29, output_size=10):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.output_size = output_size
        self.build()
    
    def build(self):
        """
        Retrieves the features from the last pool layer in the densenet pretrained model 
        and pass obtained features to LSTM network. 
        """
        input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c) # input shape


        ## Getting the pre-trained DenseNet
        densenet = tf.keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet')
        for layer in densenet.layers:
            layer.trainable = False


        ######################
        ## BUILDING THE MODEL
        ######################
        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        self.image_frame_features = TimeDistributed(densenet)(self.input_data) ## extracting the features from the image

        self.drop1 = Dropout(0.5)(self.image_frame_features)
        self.flat = TimeDistributed(Flatten())(self.drop1) ## flatten before passing on to the recurrent network

        self.sequence = LSTM(256, name='lstm')(self.flat)

        self.drop2 = Dropout(0.5)(self.sequence)
        self.dense = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.drop2)

        self.pred = Activation('softmax', name='softmax')(self.dense)

        self.model = Model(inputs = self.input_data, outputs=self.pred)

    def summary(self):
        """"Summarizes the architecture of the model.
        
        :return: returns the model architecture summary
        """
        return self.model.summary()

    def predict(self, input_batch):
        """Predicts a video
        
        :param input_batch: A batch of a sequence of frames. 
        :return: returns the predicted probailities
        """
        return self.model(input_batch)