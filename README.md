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

2. Explain how you settled on your final choice of HOG parameters.

orient = 9
pix_per_cell = 8
cell_per_block = 2

These parameters were selected similar to the examples covered in the coursework and finetuned for
this application.
I tried various combinations of parameters. RGB gave better accuracy than HSL, HSV and other color
spaces. Parameters were varied for more accurate predictions.

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG
features (and color features if you used them).

I trained a linear SVC after cell 6. The LinearSVC() function is used for the linear SVC.
svc.fit(X_train, y_train) is used for fitting the model. These were the results:

<I>Using spatial binning of: 32 and 32 histogram bins
Feature vector length: 3168
62.14 Seconds to train SVC...
Test Accuracy of SVC = 0.92</I>

For HOG classification:

<I>Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 1764
0.24 Seconds to train SVC...
Test Accuracy of SVC = 0.965
My SVC predicts: [ 0. 1. 0. 0. 1. 1. 0. 1. 0. 0.]
For these 10 labels: [ 0. 1. 0. 0. 1. 1. 0. 1. 0. 0.]</I>

For Search and classification: (accuracy of 99% is obtained.)

<I>Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 2580
0.17 Seconds to train SVC...
Test Accuracy of SVC = 0.99</I>
