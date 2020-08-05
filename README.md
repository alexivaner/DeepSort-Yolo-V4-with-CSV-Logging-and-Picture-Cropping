# Deep Sort Yolo V4 with CSV Logging and Picture Cropping

I added csv tracking logging using Pandas in DeepSort Yolo v4. The csv tracking is very useful if you want to train the csv inside GRU neural network or other kind of deep learning train. I also added person cropping so we could get each person tracked and each person frames picture one by one. This frames picture will be very useful if you want to train for example inside FlowNet.

## Original Repository
<a href="https://github.com/LeonLok/Deep-SORT-YOLOv4">LeonLok Deep-SORT-Yolov4</a>

## How to use

If you use conda, you can use my "requirement.txt" by type:
```
conda create --name <env> --file requirements.txt
```


Go to tensorflow2.0-> deep-sort-yolov4
```
cd tensorflow2.0/deep-sort-yolov4/
```
### Directory Trees of the Project

<pre>
└── <font color="#3465A4"><b>tensorflow2.0</b></font>
    ├── <font color="#3465A4"><b>deep-sort-yolov4</b></font>
    │   ├── convert.py
    │   ├── <font color="#3465A4"><b>deep_sort</b></font>
    │   │   ├── detection.py
    │   │   ├── detection_yolo.py
    │   │   ├── __init__.py
    │   │   ├── iou_matching.py
    │   │   ├── kalman_filter.py
    │   │   ├── linear_assignment.py
    │   │   ├── nn_matching.py
    │   │   ├── preprocessing.py
    │   │   ├── <font color="#3465A4"><b>__pycache__</b></font>
    │   │   │   ├── detection.cpython-36.pyc
    │   │   │   ├── detection_yolo.cpython-36.pyc
    │   │   │   ├── __init__.cpython-36.pyc
    │   │   │   ├── iou_matching.cpython-36.pyc
    │   │   │   ├── kalman_filter.cpython-36.pyc
    │   │   │   ├── linear_assignment.cpython-36.pyc
    │   │   │   ├── nn_matching.cpython-36.pyc
    │   │   │   ├── preprocessing.cpython-36.pyc
    │   │   │   ├── track.cpython-36.pyc
    │   │   │   └── tracker.cpython-36.pyc
    │   │   ├── tracker.py
    │   │   └── track.py
    │   ├── demo_csv_crop.py
    │   ├── demo_csv.py
    │   ├── <font color="#CC0000"><b>demo-csv.zip</b></font>
    │   ├── demo.py
    │   ├── IMG_3326.MOV
    │   ├── <font color="#CC0000"><b>jpg2png.zip</b></font>
    │   ├── <font color="#3465A4"><b>model_data</b></font>
    │   │   ├── coco_classes.txt
    │   │   ├── mars-small128.pb
    │   │   ├── voc_classes.txt
    │   │   ├── yolo4.h5
    │   │   ├── yolo_anchors.txt
    │   │   └── yolov4.weights
    │   ├── <font color="#3465A4"><b>output_crop</b></font>
    │   │   ├── <font color="#3465A4"><b>1</b></font>
    │   │   │   ├── <font color="#75507B"><b>frame_100.png</b></font>
    │   │   │   ├── <font color="#75507B"><b>frame_101.png</b></font>

</pre>

## If you want to try with your own video:
You should replace the path of the video with path of your video in Line 44 of code you want to run:
Line 44 of demo.py or demo_csv.py or demo_csv_crop.py

```
44|    file_path = 'IMG_3326.MOV'
```


## How to Visualize Tracking and have video output
After you <strong>Go to tensorflow2.0-> deep-sort-yolov4</strong>, :


I already put pretrained yolov4.weight, you just need to type
accordingly and run:
```
python convert.py
```
Then run demo.py:
```
python demo.py
```
You will get visualization of your video tracking in .avi format in "deep-sort-yolov4" folder:
![Result_Video](https://github.com/alexivaner/DeepSort-Yolo-V4-with-CSV-Logging-and-Picture-Cropping/blob/master/result_github/yolo%20v4.gif)

## How to Visualize Tracking and Output CSV Tracking Files
Make sure you already run convert.py before:
```
python convert.py
```
Then run demo.py:
```
python demo_csv.py
```

You will get CSV files "result_tracking.csv" something like this:
![Result_CSV](https://github.com/alexivaner/DeepSort-Yolo-V4-with-CSV-Logging-and-Picture-Cropping/blob/master/result_github/csv.png)



## How to Output CSV Files and Get the Cropped Tracking Result
Make sure you already run convert.py before:
```
python convert.py
```
Then run demo.py:
```
python demo_csv_crop.py
```

You will get cropped result in "output_crop" folder something like this:<br>
![Result_Crop1](https://github.com/alexivaner/DeepSort-Yolo-V4-with-CSV-Logging-and-Picture-Cropping/blob/master/result_github/1.gif)
![Result_Crop2](https://github.com/alexivaner/DeepSort-Yolo-V4-with-CSV-Logging-and-Picture-Cropping/blob/master/result_github/2.gif)

