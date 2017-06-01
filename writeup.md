##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_image]: ./writeup_images/hog_channel0.png
[vehicle_location]: ./writeup_images/autti.png
[vehicle_sizes]: ./writeup_images/autti_car_sizes.png
[search_example]: ./writeup_images/sliding_window_example.gif
[final_example1]: ./output_images/test1.jpg
[final_example2]: ./output_images/test2.jpg
[pipeline_example1]: ./writeup_images/Example1.png
[pipeline_example2]: ./writeup_images/Example2.png
[pipeline_example3]: ./writeup_images/Example3.png
[pipeline_example4]: ./writeup_images/Example4.png
[video1]: ./project_video.mp4


## Notes:

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####0. Code Location and Information
Code can be found inside the [src directory](https://github.com/stridera/CarND-Vehicle-Detection/tree/master/src).  All code is written with functionality separated into different Classes.  All classes have code at the bottom that test the class and can be run by running the class file directly from the command line.

Files are:

* [Image\_Utils.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Image_Utils.py) - All image manipulation and feature extraction functionality.
* [Sliding\_Window\_Search.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Sliding_Window_Search.py) - The sliding window class
* [Classifier.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Classifier.py) - The SVC classifier
* [pipeline.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/pipeline.py) - The actual pipeline which uses all the other classes to identify vehicles.
* [get_car_location_data.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/get_car_location_data.py) - A test code branch that goes through the autti and crowdui data and marks up where it discovers vehicles.  It helped determine where to have the sliding window search do its magic.

####1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG parameter extraction occurs in [Image\_Utils.py Line 84](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Image_Utils.py#L84).  I used the example from the class and created two paths, starting on line 73, that either allow you to stack all three channels together, or select a single channel.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)` and with `feature_vector=True`:

![Hog Example Image][hog_image]

####2. Explain how you settled on your final choice of HOG parameters.

For this part, I created an array of parameters and ran it through, marking up images.  I used some actual images hand selected from the video to ensure that they were working correctly.  This part took a while, so I let it go overnight and save the images with their parameters and outputs marked.  While my end result still isn't perfect, I have some ideas on how to improve it.  (Shown at end.)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier code is defined in the [Classifier](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Classifier.py) class.  This class is fairly simple and uses a Linear SVC classifier to attempt and identify the cars.  If set to train the data, it takes all the cars/non-cars data given in the project page, randomizes them, cuts a slice off for training, and then sends it through the image pipeline before attempting to fit it.

After training, we dump the trained classifier to a pickle file so we don't have to train it every time.

Finally, we have a predict function that simply runs an image through the trained classifier and returns the guess.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My first concern when implementing a window search was to figure out where the cars are usually located and the sizes of vehicles.  To figure this out, I took some of the udacity provided externally labeled data and created a heatmap of where the vehicles were labeled.  

You can see the code I used for this process in [get_car_location_data.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/get_car_location_data.py).  The process was simply going through the metadata and drawing them on the image that was zeroed out using the image size of the first image we looked at.

After processing the metadata, I was given this heatmap:

![Vehicle Locations Heatmap][vehicle_location]

Also, looking at vehicle sizes we get the following:

![Vehicle Size Graph][vehicle_sizes]

We can see that the vehicles pretty much can be seen anywhere about 20% down to the bottom.  Also, vehicle sizes are generally in a square location going from almost zero pixels all the way up to the full screen.  (Image was 1920x1200 pixels.)  

The actual sliding window code can be found in the [Sliding\_Window\_Search.py](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Sliding_Window_Search.py)

Using this, I started working on the Sliding Window Search.  My goal was to have a completely customizable and extensible search class.  You can initialize the class by giving it the search area, the min/max search window sizes, the steps, overlap, etc.  As it searches, it sends the clipped image to the validator function which is expected to return true/false.  If true, it calls the on_valid callback, otherwise it calls the invalid callback.  [Line 53](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Sliding_Window_Search.py#L53)


The actual process starts on [line 24](https://github.com/stridera/CarND-Vehicle-Detection/blob/master/src/Sliding_Window_Search.py#L24).  We start by getting the search window, the test window size, and then figuring out how many of these windows we need to fill the search area.  The next step is simply stepping through each location and running it through the test function.

As an example, I ran it through with the validator returning true for every step, and had it draw a square on each location and saved the image.  Using ImageMajick to combine them, I was given the following image illustrating the search pattern:

![Search Window Example][search_example]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Some examples
![Final Example 1][final_example1]
![Final Example 2][final_example2]

I optimized the code by using callbacks and allowing the code to dynamically validate and process the search windows as they found it.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video-processed.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the following images, you can see the pipeline as it goes through and marks all the boxes that the classifier identifies as true.  It then creates a heatmap using those squares.  I found that a threshold of about 75 worked relatively well at clearing out the false positives and leaving me with good locations of the vehicles.  It's not the best, but it works for the first round.  I then used `scipy.ndimage.measurements.label` to mark all of the identified zones and stepped through each one and drew the final results on the image.

![Pipeline Example 1][pipeline_example1]
![Pipeline Example 2][pipeline_example2]
![Pipeline Example 3][pipeline_example3]
![Pipeline Example 4][pipeline_example4]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues:
My pipeline seems to not like white cars at all.  I wanted to use more data from the kitti dataset to better train the classifier.  I have a sneaking suspicion that it's identifying dark areas as cars, and I think it's because most of the training set was dark cars.

Ideas for improvement:

* One improvement would be to send each image through the image processing tool as a whole and then 'crop' out the sub-sections of each window and send that through the classifier.  This is more difficult because you need to keep each feature set separate, cut out the search window from each window, and then ravel it together.
* Another step I wanted to do, but ran out of time, was to process each 'guessed' frame through a trained Neural Network.  I think the NN would work better, and by combining it with the speed of a SVC, it would give the best in both performance and accuracy.