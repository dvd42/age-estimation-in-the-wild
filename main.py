# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchvision import models
import argparse
import imutils
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--webcam", action='store_true',
        help="use webcam to record video and process it")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak face detections")
ap.add_argument("-img", "--images", default="images",
        help="path to where the images are")
ap.add_argument("-o", "--output", default="output",
        help="path to store output")

args = vars(ap.parse_args())


def compute_margin(box, h, w):

    startX, startY, endX, endY = box.astype("int")

    # Compute 40% margin around the face in the fram in the frame
    new_startY = startY - int((endY - startY + 1) * 0.4)
    new_endY = endY + int((endY - startY + 1) * 0.4)
    new_startX = startX - int((endX - startX + 1) * 0.4)
    new_endX = endX + int((endX - startX + 1) * 0.4)

    # Ensure that the region cropped from the original image with margin
    # doesn't go beyond the image size
    startX = max(new_startX, 1)
    startY = max(new_startY, 1)
    endX = min(new_endX, w)
    endY = min(new_endY, h)

    return startX, startY, endX, endY


def detect_faces(frame):

    frame = imutils.resize(frame, height=1000, width=1000)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (256, 256)), 1.0,
            (256, 256), (123, 117, 104))

    net.setInput(blob)
    detections = net.forward()
    cropped_faces = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        # Extract our margin indexes to crop the faces in the frame
        startX, startY, endX, endY = compute_margin(box, h, w)

        cropped_faces.append(frame[startY:endY, startX:endX])

    return frame, detections, cropped_faces


def draw_boxes(frame, detections, age, gender):


    (h, w) = frame.shape[:2]

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        # Extract our margin indexes to crop the faces in the frame
        startX, startY, endX, endY = compute_margin(box, h, w)

        color = (0, 0, 255) if gender.max(1)[1][i] == 1 else (0, 255, 0)
        text = "Age: {:.2f}".format(age[i])
        y = endY- 10 if endY - 10 > 10 else endY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7 , color, 2)


    return frame

def forward(input):

    with torch.no_grad():

        age_model.eval() # we will only run inference on this model
        gender_model.eval() # we will only run inference on this model

        age = age_model(input.to(device))
        age = F.softmax(age, dim=1)
        age = age * torch.arange(101, dtype=torch.float32, device=device)
        age = torch.sum(age, 1).to(device)

        gender = gender_model(input.to(device))
        gender = F.softmax(gender, dim=1)

    return age, gender


def build_batch(cropped_faces, frame):


    input = torch.ones(len(cropped_faces), 3, 224,224).to(device)

    for i, face in enumerate(cropped_faces):

        if face.shape[0] > 224 and face.shape[1] > 224:

            face = cv2.resize(face, (224,224))
            input[i] = torch.from_numpy(face).permute(2,0,1)

    return input


def process_frame(frame):

    # Find faces in our frame
    frame, detections, cropped_faces = detect_faces(frame)

    # If there are faces in the frame
    if cropped_faces:

        # Build our batch with faces
        input = build_batch(cropped_faces, frame)

        # Estimate age and gender with both of our models
        age, gender = forward(input)

        # Draw rectangles around the faces to indicate age and gender
        frame = draw_boxes(frame, detections, age, gender)


    return frame


if __name__ == "__main__":

    # Use gpu if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load our serialized models from disk
    net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt.txt',
            'model/res10_300x300_ssd_iter_140000.caffemodel')
    age_model = torch.load("model/age_model.pth").to(device)
    gender_model = torch.load("model/gender_model.pth").to(device)

    if args["webcam"]:

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(args["output"], 'cam.avi'),fourcc, 20.0, (1000, 750))

        # Initialize the video stream and allow the cammera sensor to warmup
        vs = cv2.VideoCapture(-1)
        time.sleep(1.0)

        while True:

            _, frame = vs.read()
            frame = process_frame(frame)

            cv2.imshow("Frame", frame)
            out.write(cv2.resize(frame, (1000, 750)))
            cv2.moveWindow("Frame", 100,50);

            # If the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()
        out.release()

    else:
        path = args["images"]
        images = sorted(os.listdir(path))

        for img in images:

            frame = cv2.imread(os.path.join(path, img))
            frame = process_frame(frame)
            
            cv2.namedWindow(img ,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(img, 1000,1000)
            cv2.imshow(img, frame)
            cv2.moveWindow(img, 100,50);
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(args["output"], img), frame)

            # If the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
