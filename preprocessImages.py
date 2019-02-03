import numpy as np
# Script written to preprocess pong images for deep Q learning.
# Crop dimensions and processing steps taken from Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5).

def preprocess(image):
    # Initial dims are 210x160x3, output dims are 80x80x1.
    
    image = image[35:195] # crop
    image = image[::2, ::2, 0] # Downsample by 2.
    image[image == 144] = 0 # Remove background.
    image[image == 109] = 0 # Remove background.
    image[image != 0] = 1 # Set paddles and ball to 1.
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    
    return image