# image_reader.py
from PIL import Image
import numpy as np


class ImageReader:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.img_gray = self.img.convert('L')
        self.img_array = np.array(self.img_gray) / 255.0

    def get_image(self):
        return self.img

    def get_image_array(self):
        return self.img_array
