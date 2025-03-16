# main.py
import pandas as pd
import numpy as np

from .image_reader import ImageReader
from .origin_detector import OriginDetector
from .axis_scale_processor import AxisScaleProcessor



def main():
    image_path = "test_img_1.png"

    # Read the image
    image_reader = ImageReader(image_path)
    img = image_reader.get_image()
    img_array = image_reader.get_image_array()

    # Detect the origin
    origin_detector = OriginDetector(img_array)
    thresh = origin_detector.detect_edges()
    origin_x, origin_y = origin_detector.find_origin(thresh)
    origin_rgb = origin_detector.get_rgb_at_origin(img, origin_x, origin_y)

    print(f"Detected Origin at (X: {origin_x}, Y: {origin_y})")
    print(f"RGB value at origin (X: {origin_x}, Y: {origin_y}): {origin_rgb}")

    # Optionally, visualize the result
    origin_detector.visualize_origin(img, origin_x, origin_y)
    axis_scale_processor = AxisScaleProcessor(image_reader)

    # Example input for cropping coordinates
    start_pixel = (origin_x, origin_y)  # Starting pixel (x, y)
    direction = 'bottom-right'  # Can be 'top-left', 'top-right', 'bottom-left', or 'bottom-right'

    print(f"Cropping image from start pixel {start_pixel} towards {direction}")

    # Process the image (crop and extract text)
    if direction == 'top-left':
        col_ix = 2
    else:
        col_ix = 1
    df = axis_scale_processor.process_image(start_pixel, direction)
    slope_df = pd.DataFrame((df[col_ix].values[:, None] - df[col_ix].values) / (df[0].values[:, None] - df[0].values))
    # slope_df = slope_df.replace([0, np.nan], 99999)

    print(df)
    print(slope_df)
    # print(mode_values, mode_indexes)


if __name__ == "__main__":
    main()
