import sys
import numpy as np
import cv2 as cv


class Parameters:

    def __init__(self,  train_path):
        self.train_path = train_path
        self.show_path = False
        self.color_path = (0, 0, 255)
        self.image_type = 'jpg'
        self.method_select_path = 'greedy'
        self.factor_amplification = 1.05
        self.crop_width = 1000
        self.crop_height = 1000
        self.answer_type = 'txt'
