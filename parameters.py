import sys
import numpy as np
import cv2 as cv


class Parameters:

    def __init__(self,  train_path):
        self.train_path = train_path
        self.image_type = 'jpg'
        self.crop_width = 1000
        self.lee_speed = 10
        self.percentage = 20
        self.crop_height = 1000
        self.answer_name = "_gt"
        self.bonus_answer_name = "_bonus_gt"
        self.answer_path = "predictions/clasic"
        self.jigsaw_answer_path = "predictions/jigsaw"
        self.predicted_answer_name = "_predicted"
        self.predicted_bonus_answer_name = "_bonus_predicted"
        self.answer_type = 'txt'
