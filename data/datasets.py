import os 
import itertools
import tensorflow as tf 
import imageio
from collections import deque
tf.enable_eager_execution()

imageio.plugins.ffmpeg.download()

"""
This model generates generator of the datasets for the Network. 

@authors : Mustapha Tidoo Yussif, Samuel Atule, Jean Sabastien Dovonon
         and Nutifafa Amedior. 
"""
class GenerateDataset(object):
    def __init__(self, filename):
        self.filename = filename

    def load_video(self):
        """Loads the specified video using ffmpeg.

        Returns:
            List[FloatTensor]: the frames of the video as a list of 3D tensors
                (channels, width, height)"""

        vid = imageio.get_reader(self.filename,  'ffmpeg')
        frames = deque()
        for i in range(0, len(vid)-1):
            image = vid.get_data(i)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            frames.append(image)
        return frames
    
    def crop_frames(self, frames):
        """
        Crops the frames of the videos around the mouth region.
        This is the part that is most important part and relevant
        to the model (where we can get the relevant features)

        :param frames: The frames in the video. 
        :return: returns the croped frames.
        """
        pass 
