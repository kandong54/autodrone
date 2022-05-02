# Model
Flower Detection using Open Images V6 and YOLOv5

## Dataset
There are many [datasets](https://paperswithcode.com/datasets?task=object-detection) in object detection, and [COCO](https://cocodataset.org/#home) is the most commonly used. However, most of these datasets don't contain flowers.

As "the largest existing dataset with object location annotations", [Open Images V6](https://storage.googleapis.com/openimages/web/factsfigures.html) contains 1.9M images for 600 object classes, including flower. 

This [notebook]((model/open_images.ipynb)5) shows how to download, explore and convert the flower subset. There are 62716 images in the training set, 6949 images in the test set, and 2336 images in the validation set. To [reduce False Positives](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results), 10% background images are added to the subset later.

The flower subset with background images in yolov5 format can be found in Google Drive AutoDrone/Datasets/open-images-v6/open-images-v6.zip .

## Algorithm
This [notebook](model/yolov5.ipynb) shows how to train, export and validate the [YOLOv5](https://github.com/ultralytics/yolov5) model. 

Considering the limited computation on drones, YOLOv5n with 4.5M FLOPs was chosen. A pretrained model was trained for 50 epochs on the lab GPU computer .

The model was then exported to tflite format with [quantization](https://www.tensorflow.org/lite/performance/post_training_quantization). The table below shows the accuracy and performance of different quantifications:

| Quantization | mAP   | Speed (s) |
| ------------ | ----- | --------- |
| Float32      | 0.36  |           |
| Float16      | 0.36  | 0.47      |
| Dynamic      | 0.36  | 0.44      |
| Int8         | 0.345 | 0.24      |

To evaluate mAPs, tflite_runtime for x86_64 was built on the lab computer. Speeds were measured on a Raspberry Pi 4 Model B 4GB with official tflite_runtime in 64-bit Pi OS.

The traning results and exported models can be found in Google Drive AutoDrone/Models/YOLOv5n 0.361/ .