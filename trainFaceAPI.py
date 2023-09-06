import cv2
import numpy as np
import os

from flask import Flask, request
from flask_cors import CORS
import json
import requests

from deepface.commons import distance as dst
from deepface.detectors import FaceDetector

from dotenv import load_dotenv

load_dotenv()


import pymongo

app = Flask(__name__)
CORS(app)

detector_name = os.getenv('detector_name')

detector = FaceDetector.build_model(detector_name)


def getImage(r):
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def saveEmbedding(embedding, name, img_path):
    mongo = pymongo.MongoClient('localhost', 27017)
    db = mongo.embeddings
    mappings= db['embeddings']
    mappings.insert_one({'person_name': name, 'encoding': embedding, 'img_path': img_path})
     
    pass

def getOldEmbeddings():
    mongo = pymongo.MongoClient('localhost', 27017)
    db = mongo.embeddings
    mappings= db['embeddings']
    data = mappings.find()
    encoded_data = dict()
    for obj in data:
        # print('[INFO] encoding :' ,obj['encoding'])
        encoded_data[obj['person_name']] = obj['encoding'][0]
    return encoded_data


def apiCall(frame):
    _, img_encoded = cv2.imencode('.jpg', frame) 
    print(f'    [Detector-Info] calling faceDetector api...')
    # logger.info("[INFO] line 85 from predtrain")
    response = requests.post(
        # url='http://127.0.0.1:12000/predict', # face detection with boundry box
        url='http://127.0.0.1:' + os.getenv('faceEncoderAPI_port')+'/predict', # face detection with boundry box
        data=img_encoded.tobytes()
    )
    # logger.info("[INFO] line 90 from predtrain")
    print(f'    [Detector-Info] faceDetector gave Response {response.status_code}...')
    data= json.loads(response.text)
    # print(f'[INFO] data : {data}')
    return data['embeddings']

@app.route("/trainFace", methods=['POST','GET'])
def trainFace():
    img_file = request.files['image']
    person_name = img_file.filename.split('.')[0]
    img_file.save(f'./faces/{person_name}')
    img_file.save(f'./trained_faces/{person_name}')
    saved_path = f'./faces/{person_name}'
    face = cv2.imread(saved_path)
    embedding = apiCall(face)
    # print(f'[INFO] Got embedding')
    saveEmbedding(embedding, person_name, saved_path)     
    return 'Success'


@app.route("/recognizeFace", methods=['POST','GET'])
def recognize_image():
    final_data = dict()
    img = getImage(request)
    _,img_encoded=cv2.imencode('.jpg',img)
    # response = requests.post(url='http://127.0.0.1:12000/predict',data=img_encoded.tobytes())
    response = requests.post(url='http://127.0.0.1:' + os.getenv('detectFace')+'/predict',data=img_encoded.tobytes())
    data = json.loads(response.text)
    boxes = data['boxes']
    confidences = data['confidence']
    query_face_encodings = np.array(data['embeddings']).tolist()
    # print(f'[INFO] len1 : {len(query_face_encoding)}')
    oldencodings = getOldEmbeddings()
    names = []
    for query_face_encoding in query_face_encodings:

        distance = []
        people_names = []
        for person_name, older_face_encodings in oldencodings.items():
            # print(f'[INFO] person name : {person_name}')
            # print(f'[INFO] Len : {len(older_face_encodings)}')
            # print(f'[INFO] Len2 : {len(query_face_encoding)}')
            dist = dst.findCosineDistance(older_face_encodings, query_face_encoding)
            distance.append(dist)
            people_names.append(person_name)
        distance =np.array(distance)
        if float(distance.min()) < float(os.getenv('recognize_threshold')):
            # print('the threshold value is',float(os.getenv('threshold')))
            arg= int(distance.argmin())
            # idd= int(Ys[arg])
            name = people_names[arg]
        else:
            name = 'Unknown'
        names.append(name)
        
    final_data['boxes'] = boxes
    final_data['names'] = names

    return json.dumps(final_data)




if __name__ == '__main__':
	app.run(debug=True,host="0.0.0.0", port=int(os.getenv('trainFaceAPI')))