#!/usr/bin/python
import cv2
import os
import sys
import numpy as np
from PIL import Image
import imutils
import argparse

# default classifiers for face detection, copied from google
cascadePath = "face_cascades/haarcascade_profileface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):
    """
    convert images to matrices
    assign label to every image according to person
    Using test data to make the machine learn this data
    Args:
            path: path to images directory

    Returns:
    matrix of images, labels
    """
    i = 0
    # image_paths = [os.path.join(path, f)
    #                for f in os.listdir(path) if not f.endswith('.sad')]
    # images = []
    # labels = []
    # for image_path in image_paths:
    #     image_pil = Image.open(image_path).convert('L')
    #     image = np.array(image_pil, 'uint8')
    #     image = imutils.resize(image, width=min(500, image.shape[1]))
    #     nbr = int(os.path.split(image_path)[1].split(
    #         ".")[0].replace("subject", ""))
    #     faces = faceCascade.detectMultiScale(image)
    cap = cv2.VideoCapture(path)
    labels, images = [], []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
        )
        for (x, y, w, h) in faces:
            images.append(frame[y: y + h, x: x + w])
            cv2.imwrite("subject-adi."+str(i)+".jpg",frame[y: y + h, x: x + w])
            i=i+1
            labels.append("adi")
            print(i)
            # DIVYANSH - the line below can be commented out
            # cv2.imshow("Adding faces to traning set",
                    #    frame[y: y + h, x: x + w])
            # cv2.imshow('win', image[y: y + h, x: x + w])
            # cv2.waitKey(100)
    return images, labels


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to images directory")
args = vars(ap.parse_args())
path = args["images"]
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))
"""
save the trained data to cont.yaml file
"""
recognizer.save("model.yaml")
