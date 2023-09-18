

# IdentityRecognition
Identifies the people by matching their faces with the existing faces in the database

![conversions_65088c906bfe9800252eb806_conversion_V4](https://github.com/skp-1997/IdentityRecognition/assets/97504177/fcc628c9-c78a-49fc-8d5c-6d8aa85cda23)


# Running the setup

Run the face detection API

```
python detectFace.py
```
Run the training API
```
python trainFaceAPI.py
```

# Training new faces

For training faces, use PostMan for sending request to training API.
<img width="888" alt="Screenshot 2023-09-18 at 10 39 01 AM" src="https://github.com/skp-1997/IdentityRecognition/assets/97504177/49314d61-a141-4709-a2df-0fe14958b4b1">


# Testing individual faces

To check whether training has been properly done, send API request using PostMan.
<img width="900" alt="Screenshot 2023-09-18 at 10 38 48 AM" src="https://github.com/skp-1997/IdentityRecognition/assets/97504177/d233fa62-57b1-4b06-b30f-ff4db8f9e7b8">


# To get results on the video file / Camera, run 

```
python livestream.py
```







