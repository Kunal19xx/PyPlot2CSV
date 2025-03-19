from .image_reader import ImageReader
from .origin_detector import OriginDetector
from .axis_scale_processor import AxisScaleProcessor
from .point_extractor import PointExtractor


def extract_data(image_path, output_file="output.csv", colour_num = 3):
    """Extract data from a Matplotlib plot image and return a DataFrame."""
    image_reader = ImageReader(image_path)
    img_array = image_reader.get_image_array()

    # Detect the origin
    origin_detector = OriginDetector(img_array)
    thresh = origin_detector.detect_edges()
    origin_x, origin_y = origin_detector.find_origin(thresh)

    print(f"Detected Origin at (X: {origin_x}, Y: {origin_y})")

    # Process axis scale (assuming implementation exists)
    axis_scale_processor = AxisScaleProcessor(image_reader)
    scale_x, scale_y = 1.0, 1.0

    # Extract points
    extractor = PointExtractor(image_reader,
                               (origin_x, origin_y),
                               (origin_detector.x_axis, origin_detector.y_axis),
                               k = colour_num)
    extractor.process_image()
    extractor.analyze_emptiness()
    extractor.plot_extracted_points(num_colors=20)

    # Save and return DataFrame
    extractor.save_to_csv(output_file)
    return extractor.df  # Return extracted data as a DataFrame
