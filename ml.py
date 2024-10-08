from deepface import DeepFace
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import pymongo

# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb+srv://mrkrishnaaggarwal:Golgappe%404777@phoenix.koa7f6e.mongodb.net/Kaksha")
db = mongo_client["Kaksha"]
collection = db["students"]  # Replace with your collection name

# Ensure the directory exists
SAVE_DIR = './folder'
os.makedirs(SAVE_DIR, exist_ok=True)

def save_image(image, path):
    cv2.imwrite(path, image)

def decode_and_save_images():
    documents = collection.find({}, {"rollNo": 1, "photograph": 1})

    for document in documents:
        rollNo = document.get("rollNo")
        encoded_photo = document.get("photograph")

        if rollNo and encoded_photo:
            try:
                # Log a portion of the base64 string for inspection
                print(f"RollNo {rollNo} - Base64 String (first 100 chars): {encoded_photo[:100]}")

                # Ensure proper padding of the base64 string
                if len(encoded_photo) % 4 != 0:
                    encoded_photo += '=' * (4 - len(encoded_photo) % 4)

                # Decode the base64 encoded image
                image_data = base64.b64decode(encoded_photo)

                # Convert bytes to a NumPy array
                nparr = np.frombuffer(image_data, np.uint8)

                # Decode the NumPy array to an image
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Define the file path
                file_path = os.path.join(SAVE_DIR, f'{rollNo}.jpg')

                # Save the image using OpenCV
                save_image(image, file_path)

            except Exception as e:
                print(f"An error occurred for rollNo {rollNo}: {e}")

decode_and_save_images()

# Initialize Flask app
app = Flask(__name__)

@app.route('/search_faces', methods=['POST'])
def search_faces():
    if 'image' in request.json:
        # Decode the base64 encoded image string
        image_data = request.json['image']
        image_bytes = base64.b64decode(image_data)
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Open the image using Pillow and convert it to RGB format
        query_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"Query Image Size: {query_img.size}")
        print(f"Query Image Mode: {query_img.mode}")

        # Convert the Pillow image to a numpy array
        query_img_array = np.array(query_img)

        # Extract faces from the query image
        query_faces = DeepFace.extract_faces(query_img_array, detector_backend='retinaface', align=True)
        print(f"Found {len(query_faces)} faces in query image.")

        # Initialize the result list
        result_list = []

        # Compare each face in the query image with faces in the saved directory
        for saved_face_file in os.listdir(SAVE_DIR):
            roll_no = os.path.splitext(saved_face_file)[0]
            saved_face_path = os.path.join(SAVE_DIR, saved_face_file)

            try:
                # Open the saved face image using Pillow
                saved_face_img = Image.open(saved_face_path).convert('RGB')
                saved_face_array = np.array(saved_face_img)
                print(f"Processing saved face file: {saved_face_file}")

                found = 0
                for query_face in query_faces:
                    query_face_array = query_face['face']
                    if query_face_array.dtype != np.uint8:
                        query_face_array = (query_face_array * 255).astype(np.uint8)

                    # Compare the faces using DeepFace
                    result = DeepFace.verify(query_face_array, saved_face_array, detector_backend='retinaface', align=True, model_name='ArcFace', distance_metric='euclidean_l2')
                    print(f"Verification result for {saved_face_file}: {result}")

                    if result['verified']:
                        found = 1
                        break

                # Append result to the list
                result_list.append([saved_face_file, found])

            except Exception as e:
                print(f"Error processing file {saved_face_file}: {e}")
                result_list.append([saved_face_file, 0])

        # Create final result dictionary
        final_result = {"rollNo": result_list}

        return jsonify(final_result)

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        return jsonify({"error": f"Processing error: {error_message}"}), 500

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=5001)
