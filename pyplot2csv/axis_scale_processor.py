# axis_scale_processor.py
from sympy import refine

from .image_reader import ImageReader
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt


class AxisScaleProcessor:
    def __init__(self, image_reader):
        self.image_reader = image_reader
        self.img = self.image_reader.get_image()
        self.img_width, self.img_height = self.img.size

    def calculate_crop_size(self, start_pixel, direction):
        """Calculates the crop width and height based on start pixel and direction."""
        start_x, start_y = start_pixel
        proportion = 1

        if direction == 'top-left':
            crop_width = int(start_x * proportion) + 5
            crop_height = start_y + 5
            start_x = int(start_x * (1 - proportion))
            start_y = 0
        elif direction == 'top-right':
            crop_width = self.img_width - start_x
            crop_height = start_y
            start_y = 0
        elif direction == 'bottom-left':
            crop_width = start_x
            crop_height = self.img_height - start_y
            start_x = 0
        elif direction == 'bottom-right':
            crop_width = self.img_width - start_x
            crop_height = self.img_height - start_y
        else:
            raise ValueError("Invalid direction. Use 'top-left', 'top-right', 'bottom-left', or 'bottom-right'.")
        print(start_x, start_y, crop_width, crop_height)
        print(self.img_width, self.img_height)
        return start_x, start_y, crop_width, crop_height

    def crop_image(self, start_pixel, direction):
        """Crops the image based on the start pixel and direction."""
        start_x, start_y, crop_width, crop_height = self.calculate_crop_size(start_pixel, direction)

        # Cropping the image using the start pixel and calculated crop size
        return self.img.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))

    def extract_text_with_coordinates(self, cropped_img):
        """Extracts text and its coordinates from the cropped image."""
        data = pytesseract.image_to_data(cropped_img, output_type=pytesseract.Output.DICT)
        text_positions = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():  # To ensure we don't include empty strings
                try:
                    # Try to convert the text to float
                    data['text'][i] = float(data['text'][i])
                    text_positions.append(
                        (data['text'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i],
                         data['conf'][i]))
                except:
                    pass  # If it can't be converted to float, skip it

        return text_positions

    def refine_extracted_text(self, text_positions, direction):
        df = pd.DataFrame(text_positions)
        df[3] = df[1] + df[3]
        df[4] = df[2] + df[4]

        # 3 for vertical
        if direction == 'top-left':
            col_ix_subject = 1
        else:
            col_ix_subject = 2

        mode_c = df[col_ix_subject].mode().iloc[0]
        df = df[(df[col_ix_subject] >= mode_c - 4) & (df[col_ix_subject] <= mode_c + 4)]

        return df

    def process_image(self, start_pixel, direction):
        """Crops the image based on the start pixel and direction, then extracts text with pixel positions."""
        cropped_img = self.crop_image(start_pixel, direction)
        text_crude = self.extract_text_with_coordinates(cropped_img)

        plt.figimage(cropped_img)
        plt.show()
        return self.refine_extracted_text(text_crude, direction)
