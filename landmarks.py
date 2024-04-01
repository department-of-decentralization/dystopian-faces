import cv2
import dlib
import np
import logging
import os
import urllib.request
import bz2

# Define groups of landmarks
LANDMARK_GROUPS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "lower_nose": list(range(30, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68))
}

LOGGING_LEVEL = logging.DEBUG


# Initialize logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

predictor_path = 'shape_predictor_68_face_landmarks.dat'
compressed_file_path = predictor_path + '.bz2'
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

# Check if the predictor file already exists
if not os.path.exists(predictor_path):
    logging.info(f"{predictor_path} not found. Downloading...")

    # Download the compressed file
    urllib.request.urlretrieve(url, compressed_file_path)
    logging.info("Download completed.")

    # Decompress the file
    with bz2.BZ2File(compressed_file_path, 'rb') as f_in:
        with open(predictor_path, 'wb') as f_out:
            f_out.write(f_in.read())
    logging.info("Decompression completed.")

    # Remove the compressed file after decompression
    os.remove(compressed_file_path)
    logging.info(f"Removed compressed file {compressed_file_path}.")
else:
    logging.info(f"{predictor_path} already exists.")
# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to add facial landmarks to an image
# Variables: 
#   line_thickness: Thickness of the lines to be drawn on the face
#   point_size: Size of the points to be drawn on the face
def add_facial_landmarks_to_image(image_data, line_thickness, point_size, color):
    logging.debug("Processing image.")
    try:
        # Convert the bytes data to a numpy array
        nparr = np.fromstring(image_data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        logging.debug("Arguments: line_thickness={}, point_size={}, color={}".format(line_thickness, point_size, color))
        # Throw an error if img is None (meaning the image could not be decoded)
        if img is None:
            raise ValueError("No image found or image data is corrupted")

        # Convert to grayscale (required for face detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)
        if len(faces) == 0:
            raise ValueError("No faces found in the image.")
        
        logging.debug("Found faces.")


        # Convert your hexadecimal color to BGR format
        color_bgr = hex_to_bgr(color)
        logging.debug(f"Color in BGR format: {color_bgr}")

        # Convert the grayscale image back to BGR to draw colored landmarks
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
                      # Draw lines for each landmark group
            for group, points in LANDMARK_GROUPS.items():
                for i in range(len(points) - 1):
                    part_a = landmarks.part(points[i])
                    part_b = landmarks.part(points[i + 1])

                    cv2.line(color_img, (part_a.x, part_a.y), (part_b.x, part_b.y), color_bgr, line_thickness)



 # Draw facial landmarks with the random color and size 3 pixels
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(color_img, (x, y), point_size, color_bgr, -1)

        # Encode the image to bytes
        _, img_encoded = cv2.imencode('.jpg', color_img)
        # Convert to a bytes object
        img_bytes = img_encoded.tobytes()

        logging.info(f"Processed image")
        return img_bytes
    except Exception as e:
        logging.error(f"Error processing: {e}")
        raise

def hex_to_bgr(color_hex):
    color_hex = color_hex.lstrip('#')  # Ensure '#' is stripped from the beginning
    # Convert the hex to RGB first
    rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    # Convert RGB to BGR
    return rgb[::-1]  # Return the reverse tuple, which is BGR
