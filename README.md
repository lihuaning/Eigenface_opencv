# Eigenface_opencv
use opencv-python, numpy to implement the eigenface in AT&amp;T dataset 

origin dataset is in **att_faces**, preprocessed dataset is in **pre_faces**. **mask.py** is to preprocess the dataset according to the eye positions which are written in the txt.

# Pre-process
to improve the training model, one way is to preprocess the images.
* set a mask according to the position of eyes
* align all the images with the mask by scaling rotating and translating

