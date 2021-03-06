import os 
import itertools
import tensorflow as tf 
import imageio
from collections import deque
import pandas as pd
import numpy as np 
from tensorflow.keras.utils import to_categorical
tf.enable_eager_execution()

imageio.plugins.ffmpeg.download()

"""
This model generates generator of the datasets for the Network. 

@authors : Mustapha Tidoo Yussif, Samuel Atule, Jean Sabastien Dovonon
         and Nutifafa Amedior. 
"""
class GenerateDataset(object):
    """Generates generator for the datasets
    
    This modles generates a generator for the datasets. This done to efficiently 
    manage space.
    
    :param: file_path: path to files/videos.
    :param directory: Path to the main directory.
    """
    def __init__(self, file_path, directory):
        self.directory = directory
        self.file_path = file_path
        self.num_samples = len(self.samples(self.get_video_files(self.file_path, self.directory)))

    def load_video(self, filename):
        """Loads the specified video using ffmpeg.

        Returns:
            List[FloatTensor]: the frames of the video as a list of 3D tensors
                (channels, width, height)"""
        
        reader = imageio.get_reader(filename,  'ffmpeg')
        
        return np.array(list(reader), dtype=np.float32)
    
    def crop_frames(self, frames):
        """
        Crops the frames of the videos around the mouth region.
        This is the part that is most important part and relevant
        to the model (where we can get the relevant features)

        :param frames: The frames in the video. 
        :return: returns the croped frames.
        """
        pass 


    def create_df(self, file_path):
        '''
        creates pandas dataframe of labels and words directories
        '''
        
        d = {}
        y_labels = []
        class_folders = []
        for ind, clss in enumerate(os.listdir(file_path)):
            y_labels.append(ind)
            class_folders.append(clss)
        
        d['directory'] = class_folders
        d['class'] = y_labels
        
        return pd.DataFrame(d)


    def get_video_files(self, file_path, directory=None):
        '''
        get video files from word class directories
        '''
        d = {}
        f = []
        
        for root, dirs, files in os.walk(file_path):
            if root.split('/')[-1] == directory:
                for file in files:
                    if file.endswith(".mp4"):
                        target_file = file.split('_')[0]
                        f.append(target_file)
                        if target_file not in d:
                            d[target_file] = []
                        d[target_file].append(os.path.join(root, file))
                    
        return d
        
    def samples(self, video_files):
        train = []
        for key, value in video_files.items():
            for file in value:
                train.append(file)
          
        return train

    def get_sample_size(self):
        return self.num_samples

    def generator(self, num_items_per_class, batch = 64):
        """Interfaces the private generator method

        :param num_items_per_class: The number of items in a categority. 
        :param batch: The batch size.
        """
        data = self.create_df(self.file_path)
        video_files = self.get_video_files(self.file_path, self.directory)
        self._generator(data, self.directory, video_files, batch) 

    def _generator(self, data, directory=None, video_files=None, BATCH_SIZE = 64):
        
        '''
        retrieves the training batch for each iteration
        '''
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256
        IMAGE_CHANNEL = 3
        NUM_FRAMES = 29
        NUM_CLASSES = 10
        
        
        train = []
        for key, value in video_files.items():
            for file in value:
                train.append(file)
        
        while True:
            # Randomize the indices to make an array
            indices_arr = np.random.permutation(len(train))
            
            for batch in range(0, len(indices_arr), BATCH_SIZE):
                # slice out the current batch according to batch-size
                current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

                # initializing the arrays, x_train and y_train
                x_train = np.empty([0, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            
                y_train = np.empty([0], dtype=np.int32)

                for i in current_batch:
                    # get an image and its corresponding color for an traffic light
                    video_frames = self.load_video(train[i])

                    # Appending them to existing batch
                    x_train = np.append(x_train, [video_frames], axis=0)
                    y_train = np.append(y_train, [ data.loc[ data['directory'] == train[i].split('/')[-1].split('_')[-2] ].values[0][0] ])
                
                y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

                yield(x_train, y_train)
        