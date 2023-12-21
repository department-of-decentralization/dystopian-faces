import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_and_compress_image(input_folder, output_folder, width=300):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        base_filename, file_extension = os.path.splitext(filename)
        output_filename = base_filename + '.jpg'
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(file_path) as img:
                # Convert PNG to JPG
                if file_extension.lower() == '.png':
                    img = img.convert('RGB')

                # Calculate the height using the aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(aspect_ratio * width)

                # Resize the image using LANCZOS resampling filter
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)

                # Compress and save the image
                img.save(output_path, "JPEG", optimize=True, quality=35)

            logging.info(f"Processed {filename} successfully.")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

# Usage
input_folder = 'landmarks-output'  # Relative path to the input folder
output_folder = 'landmarks-output-compressed'  # Relative path to the output folder
resize_and_compress_image(input_folder, output_folder)
