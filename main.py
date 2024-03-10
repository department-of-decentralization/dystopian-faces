import flask
from flask import send_file
from io import BytesIO
import functions_framework
from landmarks import add_facial_landmarks_to_image
import logging
LOGGING_LEVEL = logging.DEBUG


# Initialize logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

@functions_framework.http
def process_image(request: flask.Request):
    logging.debug("Received request")
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        logging.info("OPTIONS request")
        # Allows GET requests from origin http://ethberlin.org with
        # Authorization header
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    # Ensure there is a file in the request
    if 'file' not in request.files:
        return flask.jsonify({'error': 'no file'}), 400

    # Read the file from the request
    file = request.files['file']
    if file.filename == '':
        return flask.jsonify({'error': 'no filename'}), 400

    # Parse line_thickness and point_size from the request arguments
    line_thickness = request.args.get('line_thickness', type=int)
    point_size = request.args.get('point_size', type=int)

    try:
        # Read the image into memory
        image_data = file.read()

        # Process the image to add facial landmarks
        processed_image_data = add_facial_landmarks_to_image(image_data, line_thickness, point_size)

        if processed_image_data is None:
            return flask.jsonify({'error': 'could not process image'}), 500

        # Convert the processed image data to a BytesIO object
        processed_image_stream = BytesIO(processed_image_data)

        # Return the processed image file
        processed_image_stream.seek(0)
        response = send_file(processed_image_stream, mimetype='image/jpeg')
        response.headers.extend(headers)
        return response
    except ValueError as ve:
        # Handle specific exceptions like missing faces or corrupted images
        logging.error(f"ValueError: {ve}")
        error_response = flask.jsonify({'error': str(ve)})  # Create the response object
        error_response.status_code = 400  # Set the status code
        error_response.headers.extend(headers)  # Extend the headers of the response object
        return error_response
    except Exception as e:
        # Handle unexpected or generic exceptions
        logging.error(f"Unexpected error: {e}")
        error_response = flask.jsonify({'error': 'An unexpected error occurred'})  # Create the response object
        error_response.status_code = 500  # Set the status code
        error_response.headers.extend(headers)  # Extend the headers of the response object
        return error_response
