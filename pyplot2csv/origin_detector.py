# origin_detector.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


class OriginDetector:
    def __init__(self, img_array):
        """Initialize with an image array."""
        self.img_array = img_array.astype(np.float32)  # Ensure float32 for calculations
        self.x_axis = {}
        self.y_axis = {}

    def detect_edges(self, VERBOSE=False):
        """Apply Sobel edge detection using NumPy (fast, no PyTorch)."""

        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Apply convolution (same as conv2d in PyTorch)
        edges_x = convolve(self.img_array, sobel_x, mode="constant", cval=0.0)
        edges_y = convolve(self.img_array, sobel_y, mode="constant", cval=0.0)

        # Compute edge magnitude
        edges = np.sqrt(edges_x ** 2 + edges_y ** 2)

        # Crop borders (equivalent to PyTorch result)
        edges_cropped = edges[1:-1, 1:-1]

        # Apply binary threshold
        # _, thresh = cv2.threshold(edges_cropped, 0.1, 1, cv2.THRESH_BINARY)
        thresh = (edges_cropped > 0.1).astype(np.float32)

        # Show result
        if VERBOSE:
            plt.imshow(thresh, cmap="gray")
            plt.show()

        return thresh

    @staticmethod
    def find_non_zero_indices(input_list):
        max_val = max(input_list)
        output_li = []
        for i, val in enumerate(input_list):
            if val > max_val - 100:
                output_li.append(i)
        return output_li

    @staticmethod
    def longest_ones_indices(arr):
        """Find the longest sequence of consecutive 1s in a 1D binary array efficiently."""
        ones = np.where(arr == 1)[0]  # Get indices of ones
        if len(ones) == 0:
            return -1, -1, 0  # No 1s in the row/column

        breaks = np.where(np.diff(ones) > 1)[0]  # Find breaks in sequence
        starts = np.insert(ones[breaks + 1], 0, ones[0])  # Sequence starts
        ends = np.append(ones[breaks], ones[-1])  # Sequence ends
        lengths = ends - starts + 1  # Length of each sequence

        max_idx = np.argmax(lengths)  # Find the longest sequence
        return starts[max_idx], ends[max_idx], lengths[max_idx]

    @staticmethod
    def process_2d_array(arr, pixel_tolerance=10):
        """Process a 2D binary array to find the longest sequence of 1s for rows and columns."""
        rows_data = [OriginDetector.longest_ones_indices(row) for row in arr]  # Row-wise processing
        cols_data = [OriginDetector.longest_ones_indices(col) for col in arr.T]  # Column-wise processing (Transpose)

        df_rows = pd.DataFrame(rows_data, columns=["Start", "End", "Length"])
        df_cols = pd.DataFrame(cols_data, columns=["Start", "End", "Length"])

        max_tol_row = np.max(df_rows["Length"]) - pixel_tolerance
        max_tol_col = np.max(df_cols["Length"]) - pixel_tolerance

        df_rows_filtered = df_rows[df_rows["Length"] >= max_tol_row]
        df_cols_filtered = df_cols[df_cols["Length"] >= max_tol_col]

        return df_rows_filtered, df_cols_filtered

    def find_origin(self, thresh):
        df_rows, df_cols = self.process_2d_array(thresh)
        origin_y = max(df_cols.End)
        origin_x = min(df_rows.Start)

        w, h = self.img_array.shape

        xaxis_pixels_v = df_rows[df_rows.index > w/2].index
        yaxis_pixels_h = df_rows[df_rows.index < h/2].index

        self.x_axis['width'] = max(xaxis_pixels_v) - min(xaxis_pixels_v) + 1
        self.x_axis['length'] = max(df_rows.Length)
        self.y_axis['width'] = max(yaxis_pixels_h) - min(yaxis_pixels_h) + 1
        self.y_axis['length'] = max(df_cols.Length)


        # print("Longest sequences in rows:")
        # print(df_rows)
        # print("\nLongest sequences in columns:")
        # print(df_cols)

        return origin_x, origin_y

    def get_rgb_at_origin(self, img, origin_x, origin_y):
        img_rgb = np.array(img)
        origin_rgb = img_rgb[origin_y + 1, origin_x + 1]
        return origin_rgb

    def visualize_origin(self, img, origin_x, origin_y):
        img_rgb = np.array(img)  # Convert image to NumPy array

        plt.imshow(img_rgb)  # Display the image
        plt.scatter(origin_x + 1, origin_y + 1, s=200, c='red', marker='o')  # Draw a red circle at the origin

        plt.title("Detected Origin on Plot")
        plt.show()
