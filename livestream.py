import cv2
import requests
import json
import os
import numpy as np

from dotenv import load_dotenv

load_dotenv()

# video_path = '/Users/surajpatil/Documents/GitHub/FaceRecognition/Ricky Gervais Roasts Leonardo DiCaprio at the Golden Globes_480p.mp4'
video_path = os.getenv('video_path')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videooutput = cv2.VideoWriter(os.getenv('saved_video_path'), fourcc, 30,(854, 480))
if video_path:
    video = cv2.VideoCapture(video_path)
else:
    video = cv2.VideoCapture(0)

show = False

while True:
    try:
        # Reading frame(image) from video
        check, frame = video.read()
        if not check:
            break
        height, width = frame.shape[:2]
        _, img_encoded = cv2.imencode('.jpg', frame) 
        response = requests.post(url='http://0.0.0.0:'+ os.getenv('trainFaceAPI') + '/recognizeFace',data=img_encoded.tobytes())
        data = json.loads(response.text)
        names = data['names']
        boxes = data['boxes']
        # print(f'[INFO] Names : {names}')
        for i in range(len(boxes)):
            (x1,y1,x2,y2) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255,0,0),2) 
            label= str(names[i])
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            label_patch = np.zeros((label_height + baseline, label_width, 3), np.uint8)
            label_patch[:,:] = (0,255,255)
            labelImage= cv2.putText(label_patch, label, (0, label_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
            label_height,label_width,_= labelImage.shape
            if y1-label_height< 0:
                y1=label_height
            if x1+label_width> width:
                x1=width-label_width
            frame[y1-label_height:y1,x1:x1+label_width]= labelImage    
        if show == True:
            print(f'[INFO] Show frame')
            cv2.imshow('Output', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            # print(f'[INFO] Writing frame')
            videooutput.write(frame)       
    except Exception as e:
        print(e, '@@@@errorReport')

    

video.release()
cv2.destroyAllWindows()