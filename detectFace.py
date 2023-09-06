import os

from deepface.detectors import FaceDetector
import cv2
import time
import numpy as np

from flask import Flask, request, Response, send_file, jsonify
from flask_cors import CORS
import json

import encodeFace

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# detector_name = "mtcnn"
print(f'[INFO] detecting face...')
detector_name = os.getenv('detector_name')

print(f'[INFO] Detector name : {detector_name}')
# img = cv2.imread(img_path)

detector = FaceDetector.build_model(detector_name)

def getImage(r):
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

@app.route("/predict/ping", methods=['POST','GET'])
def ping():
	return str('ready')


@app.route("/predict", methods=['POST','GET'])
def detect_face():
    global detector_name
    global detector
    # detector_name = os.getenv('detector_name')
    # detector = FaceDetector.build_model(detector_name)
    img = getImage(request)
    # detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface
    obj = FaceDetector.detect_faces(detector, detector_name, img)
    data = dict()
    boxes = []
    confidence = []
    embeddings = []
    for i in range(len(obj)):
        if obj[i][2] > float(os.getenv('face_threshold')):
            boxes.append([obj[i][1][0],obj[i][1][1],obj[i][1][0]+obj[i][1][2],obj[i][1][1]+obj[i][1][3]])
            confidence.append(obj[i][2])
            embedding = encodeFace.embedd_face(obj[i][0])
            embeddings.append(embedding)

    '''
    for i in range(len(obj)):
        cv2.rectangle(img, (obj[i][1][0],obj[i][1][1]),(obj[i][1][0]+obj[i][1][2],obj[i][1][1]+obj[i][1][3]),(255,0,0),2 )
    '''
    data["boxes"] = boxes
    data["confidence"] = confidence
    data['embeddings'] = embeddings
    return json.dumps(data)

if __name__ == '__main__':
	app.run(debug=True,host="0.0.0.0", port=int(os.getenv('detectFace')))