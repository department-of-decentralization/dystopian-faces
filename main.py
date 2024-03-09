import flask
from flask import request, send_file
from io import BytesIO
import functions_framework
from landmarks import add_facial_landmarks_to_image

@functions_framework.http
def process_image(request: flask.Request):
    # Ensure there is a file in the request
    if 'file' not in request.files:
        return flask.jsonify({'error': 'no file'}), 400

    # Read the file from the request
    file = request.files['file']
    if file.filename == '':
        return flask.jsonify({'error': 'no filename'}), 400

    try:
        # Read the image into memory
        image_data = file.read()

        # Process the image to add facial landmarks
        processed_image_data = add_facial_landmarks_to_image(image_data)

        if processed_image_data is None:
            return flask.jsonify({'error': 'could not process image'}), 500

        # Convert the processed image data to a BytesIO object
        processed_image_stream = BytesIO(processed_image_data)

        # Return the processed image file
        processed_image_stream.seek(0)
        return send_file(processed_image_stream, mimetype='image/jpeg')

    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500