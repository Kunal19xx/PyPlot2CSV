# origin_detector.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

class OriginDetector:
    def __init__(self, img_array):
        self.img_array = img_array
        self.img_tensor = torch.tensor(self.img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def detect_edges(self):
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        edges_x = F.conv2d(self.img_tensor, sobel_x, padding=1)
        edges_y = F.conv2d(self.img_tensor, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edges_cropped = edges.squeeze().detach().numpy()[1:-1, 1:-1]
        _, thresh = cv2.threshold(edges_cropped, 0.1, 1, cv2.THRESH_BINARY)
        plt.imshow(thresh)
        plt.show()
        return thresh

    @staticmethod
    def find_non_zero_indices(input_list):
        max_val = max(input_list)
        output_li = []
        for i, val in enumerate(input_list):
            if val > max_val-100:
                output_li.append(i)
        return output_li

    def find_origin(self, thresh):
        mode_columns = [np.sum(thresh[i, :]) for i in range(thresh.shape[0])]
        mode_rows = [np.sum(thresh[:, j]) for j in range(thresh.shape[1])]
        mode_columns = self.find_non_zero_indices(mode_columns)
        mode_rows = self.find_non_zero_indices(mode_rows)
        origin_y = max(mode_columns)
        origin_x = min(mode_rows)

        return origin_x, origin_y

    def get_rgb_at_origin(self, img, origin_x, origin_y):
        img_rgb = np.array(img)
        origin_rgb = img_rgb[origin_y + 1, origin_x + 1]
        return origin_rgb

    def visualize_origin(self, img, origin_x, origin_y):
        img_rgb = np.array(img)
        img_with_origin = img_rgb.copy()
        cv2.circle(img_with_origin, (origin_x + 1, origin_y + 1), 10, (255, 0, 0), -1)
        plt.imshow(cv2.cvtColor(img_with_origin, cv2.COLOR_BGR2RGB))
        plt.title("Detected Origin on Plot")
        plt.show()
