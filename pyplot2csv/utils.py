import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

def adjust_brightness(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_grid_lines(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    return lines



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


def process_2d_array(arr, pixel_tolerance = 10):
    """Process a 2D binary array to find the longest sequence of 1s for rows and columns."""
    rows_data = [longest_ones_indices(row) for row in arr]  # Row-wise processing
    cols_data = [longest_ones_indices(col) for col in arr.T]  # Column-wise processing (Transpose)

    df_rows = pd.DataFrame(rows_data, columns=["Start", "End", "Length"])
    df_cols = pd.DataFrame(cols_data, columns=["Start", "End", "Length"])

    max_tol_row = np.max(df_rows["Length"]) - pixel_tolerance
    max_tol_col = np.max(df_cols["Length"]) - pixel_tolerance

    df_rows_filtered = df_rows[df_rows["Length"] >= max_tol_row]
    df_cols_filtered = df_cols[df_cols["Length"] >= max_tol_col]

    return df_rows_filtered, df_cols_filtered

