#
# If need will be to convert JPG to TIFF or vice versa this tool can be used.
#


import cv2

def convert_tiff_to_jpg(input_path, output_path):
    """
    Loads a TIFF image and saves it as a JPG image using OpenCV.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the output JPG file.
    """
    # Step 1: Load the TIFF image
    tiff_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if tiff_image is None:
        print(f"Error: Unable to load TIFF image from {input_path}")
        return

    print(f"Successfully loaded TIFF image with shape: {tiff_image.shape}")

    # Step 2: Save the image as a JPG file
    success = cv2.imwrite(output_path, tiff_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    if success:
        print(f"Successfully saved JPG image to {output_path}")
    else:
        print(f"Error: Failed to save JPG image to {output_path}")
      
def convert_jpg_to_tiff(input_path, output_path):
    """
    Loads a JPG image and saves it as a TIFF image using OpenCV.

    Args:
        input_path (str): Path to the input JPF file.
        output_path (str): Path to save the output TIFF file.
    """
    # Step 1: Load the TIFF image
    jpg_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if jpg_image is None:
        print(f"Error: Unable to load JPG image from {input_path}")
        return

    print(f"Successfully loaded JPG image with shape: {jpg_image.shape}")

    # Step 2: Save the image as a Tiff file convestion based on output_path file extension
    success = cv2.imwrite(output_path, jpg_image])
    
    if success:
        print(f"Successfully saved JPG image to {output_path}")
    else:
        print(f"Error: Failed to save JPG image to {output_path}")

# Example usage:
# Replace 'input.tiff' with the path to your TIFF file
# Replace 'output.jpg' with the desired output JPG file path
input_tiff_path = "input.tiff"
output_jpg_path = "output.jpg"

input_jpg_path = "input.jpg"
output_tiff_path = "output.tiff"

convert_tiff_to_jpg(input_tiff_path, output_jpg_path)
convert_jpg_to_tiff(input_jpg_path, output_tiff_path):
