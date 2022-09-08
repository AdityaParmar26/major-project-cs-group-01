import argparse
import glob
from multiprocessing.spawn import import_main_path
import os
import time
import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from background_subtraction import background_subtraction
from detect_people import detect_people
# TODO
# break to frame process

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--videos", required=True, help="path to videos directory")
    # run script -v path1,path2,path3
    args = vars(ap.parse_args())
    path = args["videos"]
    for f in os.listdir(path):
        list_of_videos = glob.glob(os.path.join(os.path.abspath(path), f))
        print(os.path.join(os.path.abspath(path), f))
        print(list_of_videos)
        for video in list_of_videos:
                camera = cv2.VideoCapture(os.path.join(path, video))
                grabbed, frame = camera.read()
                # print(frame.shape)
                # width 800 kyu set kari HYPERPARAMETER
                frame_resized = imutils.resize(frame, width=min( 800,frame.shape[1]))
                frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                # print(frame_resized.shape)
                # HYPERPARAMETER
                # defining min cuoff area
                min_area = (3000 / 800) * frame_resized.shape[1]
                while True:
                    # starttime = time.time()
                    previous_frame = frame_resized_grayscale
                    grabbed, frame = camera.read()
                    # Printing frame for debugging purpose
                    if not grabbed: break
                    cv2.imshow("Frame",previous_frame)
                    cv2.waitKey(200)
                    # end
                    frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
                    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                    temp = background_subtraction(previous_frame, frame_resized_grayscale, min_area)
                    if temp == 1:
                        frame_processed = detect_people(frame_resized)
                        cv2.imshow("Frame Selected",frame_processed)
                        cv2.waitKey(200)
                    else:
                        print("a frame dropped")



                break