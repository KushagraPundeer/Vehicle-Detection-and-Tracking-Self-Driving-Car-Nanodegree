# **Vehicle Detection Project** 

## Writeup

The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of
images and train a classifier Linear SVM classifier, Apply a color transform and append binned color
features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full
project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and
follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Images

### Image-1
<img src="images/image-1.jpg" width="850">

### Image-2
<img src="images/image-2.jpg" width="850">

### Image-3
<img src="images/image-3.jpg" width="850">

### Image-4
<img src="images/image-4.jpg" width="850">

### Image-5
<img src="images/image-5.jpg" width="850">

## RUBRIC
REFERENCE: Udacity coursework and Quizzes (Vehicle Detection)

## Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third and fourth code cell of the IPython notebook under sckitlearn
HOG extraction. I started by reading in all the `vehicle` and `non-vehicle` images. Here is an
example of one of each of the `vehicle` and `non-vehicle` classes:

<img src="images/image-car-noncar.jpg" width="850">

Used get_hog_features() to obtain HOG features.
