#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
import pandas as pd
import os
import shutil

warnings.filterwarnings('ignore')

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = 'IMG_3326.MOV'
    dfObj = pd.DataFrame(columns = ['frame_num' , 'track', 'cx' , 'cy','w','h','track_temp'])
    dfObjDTP = pd.DataFrame(columns = ['filename' , 'frame_num' , 'bb1', 'bb2' , 'bb3','bb4','track','track_temp','Height'])


    if asyncVideo_flag :
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        if tracking:
            features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        if tracking:
            # Call the tracker

            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()

                #Ini buat cropping gambar per frame

                #cropped_image = frame[int(bbox[1]):int(bbox[1])+(int(bbox[3])-int(bbox[1])),int(bbox[0]):int(bbox[0])+(int(bbox[2])-int(bbox[0]))]
                cropped_image = frame[int(bbox[1]):int(bbox[1])+256,int(bbox[0]):int(bbox[0])+128]
                # cropped_image = frame[2:5,6:10]

                # Matiin atau comment biar ga ada box putih
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                #
                # cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                #             1.5e-3 * frame.shape[0], (0, 255, 0), 1)


                # print(cropped_image)
                dirname = "output_crop/{}".format(track.track_id)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if (cropped_image.size==0):
                    continue
                else:
                    writeStatus=cv2.imwrite("output_crop/{}/frame_{}.png".format(track.track_id, frame_index), cropped_image)
                    print("output_crop/{}/frame_{}.png".format(track.track_id, frame_index))


                # Write CSV
                dfObj=dfObj.append(pd.Series([frame_index, track.track_id,
                                              int(bbox[0]) , int(bbox[1]),
                                              int(bbox[2])-int(bbox[0]),
                                              int(bbox[3])-int(bbox[1]),
                                              track.time_since_update], index=dfObj.columns ), ignore_index=True)

                dfObjDTP=dfObjDTP.append(pd.Series([file_path,frame_index,int(bbox[0]),
                                                    int(bbox[1]),int(bbox[2]),int(bbox[3]),
                                                    track.track_id,track.time_since_update,
                                                    int(bbox[3])-int(bbox[1])],
                                                    index=dfObjDTP.columns ), ignore_index=True)



        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"

            #Matiin atau comment biar ga ada box putih di crop image
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            # if len(classes) > 0:
            #     cls = det.cls
            #     cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
            #                 1.5e-3 * frame.shape[0], (0, 255, 0), 1)

        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    dfObj =  dfObj.sort_values(["track", "frame_num"], ascending = (True, True))
    dfObj.to_csv(r'result_temp.csv', index = False)
    dfObjDTP =  dfObjDTP.sort_values(["track", "frame_num"], ascending = (True, True))
    dfObjDTP.to_csv(r'result_temp_dtp.csv', index = False)
    convert_to_final()
    cv2.destroyAllWindows()

def convert_to_final():
    data = pd.read_csv("result_temp.csv")
    data_dtp = pd.read_csv("result_temp_dtp.csv")

    data['diff'] = data['track'].diff()
    # Yields a tuple of index label and series for each row in the dataframe
    a=0
    list_of_data=[]
    for (index_label, row_series) in data.iterrows():
        if(row_series.values[6]==0 and row_series.values[7]==0):
            a=a+1
            list_of_data.append(a)


        elif(row_series.values[6]==0 and row_series.values[7]==1):
            a=1
            list_of_data.append(a)


        elif(row_series.values[6]==1 and row_series.values[7]==0):
            a=a+1
            list_of_data.append(a)
            a=0

        else:
            a=a+1
            list_of_data.append(a)

    data.insert(6, "track_length",list_of_data)
    del data['diff']
    del data['track_temp']
    data_dtp.insert(7, "detection_length",list_of_data)
    del data_dtp['track_temp']




    #Labeled dan Requires_Features masih ga tau dibuat apa jadi disesuaikan dengan yg asli di nol kan semua
    data.insert(7, "labeled",0)
    data.insert(8, "requires_features",0)

    data.to_csv(r'result_tracking.csv', index = False)
    data_dtp.to_csv(r'result_tracking_dtp.csv', index = False)

if __name__ == '__main__':
    main(YOLO())
