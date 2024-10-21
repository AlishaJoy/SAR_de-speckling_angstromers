import numpy as np
from PIL import Image
import os

def npy_to_png(npy_array, output_path):
    """
    Converts a numpy array into a PNG image without modifying its data where possible.
    If the data is in float mode, it normalizes it to uint8 and then saves the image.
    
    Parameters:
    - npy_array: numpy.ndarray
        The input numpy array representing the image data.
    - output_path: str
        The path to save the PNG image.
    """
    # Handle floating-point arrays by normalizing to [0, 255]
    if npy_array.dtype == np.float32 or npy_array.dtype == np.float64:
        print(f"Converting float data to uint8 for {output_path}")
        # Normalize to [0, 255] and convert to uint8
        npy_array = (255 * (npy_array - np.min(npy_array)) / np.ptp(npy_array)).astype(np.uint8)
    
    # Ensure the input numpy array is 2D (grayscale) or 3D (RGB or RGBA)
    if npy_array.ndim == 2:
        img = Image.fromarray(npy_array, mode='L')  # Grayscale
    elif npy_array.ndim == 3 and npy_array.shape[2] in [3, 4]:
        img = Image.fromarray(npy_array)  # RGB or RGBA
    else:
        raise ValueError(f"Unsupported numpy array shape: {npy_array.shape}. Expected 2D (grayscale) or 3D (RGB/RGBA).")

    # Save the image as PNG
    img.save(output_path)
    print(f"Image successfully saved at {output_path}")


def convert_npy_to_png_batch(npy_folder, png_folder):
    """
    Converts all .npy files in a specified folder to PNG format and saves them in another folder.

    Parameters:
    - npy_folder: str
        The path to the folder containing .npy files.
    - png_folder: str
        The path to the folder where the .png images will be saved.
    """
    # Check if the input folder exists
    if not os.path.exists(npy_folder):
        raise FileNotFoundError(f"Input folder '{npy_folder}' does not exist.")
    
    # Create the output folder if it doesn't exist
    os.makedirs(png_folder, exist_ok=True)

    # Iterate through the .npy files in the directory
    for npy_file in os.listdir(npy_folder):
        if npy_file.endswith(".npy"):
            npy_path = os.path.join(npy_folder, npy_file)
            
            try:
                # Load the .npy file
                npy_data = np.load(npy_path)

                # Construct the output PNG file path
                png_file_name = npy_file.replace(".npy", ".png")
                png_path = os.path.join(png_folder, png_file_name)

                # Convert and save as PNG
                npy_to_png(npy_data, png_path)
            
            except Exception as e:
                print(f"Error processing {npy_file}: {e}")


if __name__ == "__main__":
    # Define the paths to your input and output folders
    npy_folder = "I:\sar_image\datasets_sentinal1\inputimages"  # Update with your .npy file folder path
    png_folder = "I:\sar_image\datasets_sentinal1\outputimages"  # Update with your target folder for .png files

    # Perform the conversion
    convert_npy_to_png_batch(npy_folder, png_folder)



