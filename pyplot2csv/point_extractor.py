import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import product
from .image_reader import ImageReader


class PointExtractor:
    def __init__(self, image_reader: ImageReader, center_: tuple, axis_info_: tuple):
        """Initialize with an ImageReader instance."""
        self.center_ = center_
        self.image_reader = image_reader
        self.or_x , self.or_y = center_
        self.x_axis, self.y_axis = axis_info_

        x_st = self.or_x + self.y_axis['width']
        x_en = x_st + self.x_axis['length'] - self.y_axis['width'] - 10
        y_en = self.or_y - self.x_axis['width']
        y_st = y_en - self.y_axis['length'] + self.x_axis['width'] + 10
        self.img_array = np.array(
            self.image_reader.get_image()
        )[y_st:y_en, x_st:x_en, :3]  # Ensure RGB only

        self.df = None  # DataFrame to store extracted points
        self.k = self.auto_select_k()
        # plt.figimage(self.img_array )
        # plt.show()

    @staticmethod
    def create_diagonal_colour_points(gap = 5):
        base_values = np.arange(0, 256)
        offsets = np.array(list(product(range(-gap, gap+1), repeat=3)))  # All (dx, dy, dz) variations

        # Generate all points using broadcasting
        points = (base_values[:, None, None] + offsets).reshape(-1, 3)
        return points

    def auto_select_k(self):
        """Determine the best k using the Elbow Method (fast)."""
        h, w, c = self.img_array.shape
        reshaped = self.img_array.reshape(-1, c)  # Flatten image

        wcss = []  # Store inertia values for each k
        k_values = list(range(2, 8))  # Fixed range: 2 to 7

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(reshaped)
            wcss.append(kmeans.inertia_)  # Store WCSS (sum of squared distances)

        # Find the elbow point (fast heuristic: max 2nd derivative)
        k_opt = self.find_elbow_point(k_values, wcss)

        print(f"Auto-selected k = {k_opt} (Elbow Method)")
        return k_opt

    def find_elbow_point(self, k_values, wcss):
        """Find the optimal k using the second derivative method."""
        diffs = np.diff(wcss)  # First derivative
        second_diffs = np.diff(diffs)  # Second derivative
        elbow_index = np.argmin(second_diffs) + 1  # Find highest curvature
        return k_values[elbow_index]


    def apply_kmeans(self):
        """Apply K-Means clustering to reduce colors."""
        h, w, c = self.img_array.shape
        reshaped = self.img_array.reshape(-1, c)  # Flatten image

        # Apply K-Means
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(reshaped)
        centroids = kmeans.cluster_centers_.astype(np.uint8)

        # Replace pixels with corresponding cluster centroids
        self.img_array =  centroids[labels].reshape(h, w, c)


    def process_image(self):
        """Processes the image and stores results in a DataFrame using a 2D approach."""

        # self.auto_select_k()
        self.apply_kmeans()
        h, w, c = self.img_array.shape  # Image dimensions

        # Convert to structured array without flattening
        structured_pixels = np.core.records.fromarrays(
            [self.img_array[:, :, 0], self.img_array[:, :, 1], self.img_array[:, :, 2]],
            names="r,g,b"
        )

        # Find the most common RGB
        unique_pixels, counts = np.unique(structured_pixels, return_counts=True)
        most_common_rgb = unique_pixels[np.argmax(counts)]

        # Create mask directly on the 2D array (avoid flattening)
        mask = structured_pixels != most_common_rgb
        # Extract filtered coordinates & RGBs
        filtered_y, filtered_x = np.where(mask)  # Get (y, x) indices of non-background pixels
        filtered_rgb = structured_pixels[mask]  # Extract only valid RGB values

        # Convert RGB tuples to string for column names
        rgb_keys = np.array([f"({r},{g},{b})" for r, g, b in filtered_rgb])

        # Create DataFrame
        df = pd.DataFrame({"Y": filtered_y, "X": filtered_x, "RGB": rgb_keys})

        # Pre-filter RGBs that appear in <1% of total image pixels
        rgb_counts = df["RGB"].value_counts()

        # Pivot table (faster, as we start from a smaller dataset)
        df = df.groupby(["X", "RGB"])["Y"].apply(list).unstack(fill_value=[])

        # Store cleaned DataFrame
        self.df = df.reset_index()

        print(f"Most common RGB removed: {tuple(most_common_rgb)}")
        # print(f"Dropped {len(rgb_counts) - len(keep_rgb)} rare RGB columns early.")

    def save_to_csv(self, output_path="output.csv"):
        """Saves the DataFrame to a CSV file."""
        if self.df is not None:
            renamed_df = self.df.rename(
                columns=lambda col: col.replace("(", "").replace(")", "").replace(",", "_") if col != "X" else col)
            renamed_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path} with renamed columns.")
        else:
            print("No data to save. Run process_image() first.")

    def analyze_emptiness(self):
        """Analyzes how many columns have mostly empty ([]) values."""
        if self.df is None:
            print("No data available. Run process_image() first.")
            return

        # Calculate emptiness after dropping rare RGB columns
        empty_counts = self.df.iloc[:, 1:].map(lambda x: len(x) == 0).sum()
        total_rows = len(self.df)
        empty_ratio = empty_counts / total_rows

        # Show top 5 emptiest columns
        print("\nTop 5 emptiest columns:")
        print(empty_ratio.sort_values(ascending=False).head(10))

    def plot_extracted_points(self, num_colors=5):
        """Plots extracted pixel points using RGB colors from column names."""
        if self.df is None:
            print("No data available. Run process_image() first.")
            return

        # Select top `num_colors` non-empty columns
        non_empty_columns = self.df.iloc[:, 1:].map(len).sum()
        top_columns = non_empty_columns.sort_values(ascending=False).head(num_colors).index

        clr_data = []
        for col in top_columns:

            x_values = self.df["X"]
            y_values = self.df[col]

            # Convert column name to RGB tuple
            rgb = tuple(map(int, col.strip("()").split(",")))

            color = np.array(rgb) / 255  # Normalize for matplotlib
            clr_data.append(rgb)

            # Plot non-empty points
            for y_list, x in zip(y_values, x_values):
                if y_list:
                    plt.scatter([x] * len(y_list), y_list, color=color, label=col, s=1)

        plt.gca().invert_yaxis()
        plt.title("Extracted Points Colored by RGB")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        # print(clr_data)

        x, y, z = zip(*clr_data)
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot in 3D
        ax.scatter(x, y, z, c='r', marker='o')

        # Labels
        ax.set_xlabel('R Label')
        ax.set_ylabel('G Label')
        ax.set_zlabel('B Label')

        # Show the plot
        plt.show()
