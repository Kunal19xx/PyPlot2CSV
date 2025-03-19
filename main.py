# from pyplot2csv.image_reader import ImageReader
# from pyplot2csv.origin_detector import OriginDetector
# from pyplot2csv.axis_scale_processor import AxisScaleProcessor
# from pyplot2csv.point_extractor import PointExtractor
#
#
# def main():
#     image_path = "tests/test_img_3.png"
#
#     # Read the image
#     image_reader = ImageReader(image_path)
#     img_array = image_reader.get_image_array()
#     img = image_reader.get_image()
#
#     # Detect the origin
#     origin_detector = OriginDetector(img_array)
#     thresh = origin_detector.detect_edges()
#     origin_x, origin_y = origin_detector.find_origin(thresh)
#
#
#
#     # Process axis scale
#     axis_scale_processor = AxisScaleProcessor(image_reader)
#     scale_x, scale_y = 1.0, 1.0  # Assume scale detection logic exists
#
#     # Extract points
#     extractor = PointExtractor(image_reader, (origin_x, origin_y),(origin_detector.x_axis, origin_detector.y_axis))
#     extractor.process_image()
#     extractor.save_to_csv("output.csv")
#     extractor.analyze_emptiness()
#     extractor.plot_extracted_points(num_colors=20)
#
#     print("Point extraction complete.")
#
# if __name__ == "__main__":
#     main()


from pyplot2csv import extract_data

def main():
    image_path = "tests/test_img_2.png"
    output_file = "output.csv"

    # Extract data and save to CSV
    df = extract_data(image_path, output_file, 1)

    # Display extracted data preview
    print(df.head())
    print(f"Data extracted and saved to {output_file}")

if __name__ == "__main__":
    main()