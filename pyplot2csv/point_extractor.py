import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

class PointExtractor:
    def __init__(self, img_array, origin_x, origin_y, scale_x, scale_y, target_color=(255, 0, 0)):
        self.img_array = img_array
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.target_color = np.array(target_color)

    def extract_points(self):
        img_rgb = cv2.cvtColor(self.img_array, cv2.COLOR_GRAY2RGB)
        mask = np.all(img_rgb == self.target_color, axis=-1)
        points = np.column_stack(np.where(mask))

        x_values = (points[:, 1] - self.origin_x) * self.scale_x
        y_values = (self.origin_y - points[:, 0]) * self.scale_y
        df = pd.DataFrame({'X': x_values, 'Y': y_values})

        return df

    def save_to_csv(self, df, filename="output.csv"):
        df.to_csv(filename, index=False)
        print(f"Saved extracted points to {filename}")

    def visualize_points(self, df):
        plt.scatter(df['X'], df['Y'], s=5, color='red')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Extracted Data Points")
        plt.show()
