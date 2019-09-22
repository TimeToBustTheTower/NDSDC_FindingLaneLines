# **Finding Lane Lines on the Road** 

## Motivation

### The project was the initiation of getting into the detail of implementing a key procedure of self driving cars. The goal is to illustrate recognizing, evaluating and highlighting lanes in front of a self driving vehicle.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Summarise the methodology used to obtain the result
* Highlight limitations and possible improvements


[//]: # (Image References)

[image1]: ./test_images_output/Collection_InputImages.jpg "Input Images"
[image2]: ./test_images_output/Collection_GrayImages.jpg "Gray Images"
[image3]: ./test_images_output/Collection_HSLDetectedLines.jpg "Line Detection"
[image4]: ./test_images_output/Collection_CannyMask.jpg "Filtered and Canny Masked"
[image5]: ./test_images_output/Collection_FinalResult.jpg "Final Hough Line averaged Lanes"

[//]: # (Video References)

[image6]: ./test_videos_output/solidWhiteRight.jpg "WhiteRightVideo"

---

**Images provided for the Project: Finding Lane Lines**

Following images were provided from Udacity in the "test_images" folder.
![image1]

## Procedure to make the Pipeline

### 1. Extracting White and Yellow Lanes from the Input Images

Based on my previous knowledge in photography, I realised the importance of image filtering. The selection of the filter is strongly dependant on the purpose of feature extraction from the intended image. The objective here being: clearly identifying the lanes and discarding the rest of the entities of the images. The OpenCV library, as suggested, has been used extensively for this purpose.

The lanes are clearly identified by their colour and light intensity. Hence, I opted for using the HSL. (L standing for Lightness: More material on this topic under "Using Rapid Tables for Color Definition" on the internet).

After converting the images to HSL, the focus is made on isolating the image only to the yellow and white entities in the image, which are primarily the lane colours.

![image3]

### 2. Converting Extracted Line Images to Grayscale.

The array(or rather list) of images are then converted into grayscale images.

![image2]

### 3. Canny Edge Detection and Masking of Region of Interest

The processed images and their edges are blurred for better recognition through the Gaussian Bluring filter with the appropriate gaussian kernel. [parameter selected for kernel: 5]

The focus is then made to mask the predominant region of the lanes in question. Region is set through a parameterised polygon, which is dependant on the pixel size of the images.

Upon masking the image, the Canny Edge Detection routine is used food good detection of only existent edges. Hence, the image clearly illustrates the detected lanes, which could be used further for processing and interpretations.

![image4]

### 4. Lanes using Hough transformation, averaging and extrapolation

The Hough transformation with its needed set of parameters are set through various trial and error attempts. The final set of parameters used for the Hough Transformation are:

* rho – Distance resolution of the accumulator in pixels: 1
* theta – Angle resolution of the accumulator in radians: np.pi/180
* threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes: 20
* minLineLength – Minimum line length. Line segments shorter than that are rejected: 20
* maxLineGap – Maximum allowed gap between points on the same line to link them: 300

***Averaging of the calculated hough lines into lanes and extrapolation of the lanes***

Therein, as the Bard would tell, lies the rub! It takes quite a lot of perseverance and data sorting to come up with a concept on how to go about implementing a procedure. The idea being to finally have two distance lanes (left and right) superimposed on the lanes shown in the images.

There are many methods to go about this. I myself had to discard two methods, simply because of my limitation in programming skills. Having said that, I would be coming back to this project with some more ideas and programming tricks which are available extensively in the internet to learn from various experts in the field of computer vision.

The method chosen here is simply to:
* Sort the hough lines between left and right. This is easily done by sorting them according to their slopes. (negative being left and positive being right).
* Grouping up the hough lines into lanes based on their length on either side, thereby scaling them based on their lengths.
* Extrapolating the lanes through a set distance in the image, since the images clearly have a fixed position camera and size. This makes the task quite easy to set an assumption that the extrapolated lanes would not veer off on the superimposed lanes.

The results show a functioning and qualified pipeline for the lanes on static images.

![image5]

### 5. Implementing on Video Files

The video files are processed using the moviepy module in python. Since Markdown does not permit inclusion of videos, the link to the output videos are available on the following GitHub Link:
https://github.com/TimeToBuzzTheTower/NDSDC_FindingLaneLines/tree/master/test_videos_output

***The optional challenge.mp4 video output doesnt seem work. The error message during evaluation can be reviewed in the Jupyter notebook. Issue would be evaluated and corrected at a later date due to past submission due date.***


---
## Next steps towards possible improvements and upgrades

An adaptive method towards determining the fixed parameters for the filtering processes would be needed if this were needed to be applied in actual field applications. Adaptive parameters needed would be for:

* Extracting lanes through colour detection (currently only set for thresholds of yellow and white).
* Automating filtering of unexpected dirt objects which could come across the camera.
* HSL parameter adaptive towards light sensitivity through shadows and direct sunlight or street light.
* Region of interest which is independent of image size and position of camera which is recording the image.

---
[GitHub Link]
(https://github.com/TimeToBustTheTower/NDSDC_FindingLaneLines)
