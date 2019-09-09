#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math
import statistics

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    xdimension = img.shape[1]
    ydimension = img.shape[0]
    
    array_slopes_intercept = np.zeros(shape=(len(lines),6))
    #arrayslopes = np.zeros(shape=(len(lines),1))
    
    for counter,line in enumerate(lines):
        for x1,x2,y1,y2 in line:
            slope = (y2-y1)/(x2-x1) # y = mx +c
            c = y1-x1*slope # c = y-mx
            array_slopes_intercept[counter] = [slope,c,x1,x2,y1,y2]
    
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    left_x = 0
    left_y = 0
    right_x = 0
    right_y = 0
    
    for line in array_slopes_intercept:
        if(line[0] > 0.5 and line[0] < 0.9):
            left_slopes.append(line[0])
            left_x+= x1 + x2
            left_y+= y1 + y2
        elif(line[0] < -0.5 and line[0] > -0.9):
            right_slopes.append(line[0])
            right_x+= x1 + x2
            right_y+= y1 + y2
    
    if len(left_slopes)!=0:
        left_slopes_avg = np.sum(left_slopes)/len(left_slopes)
        left_x_avg = int(left_x/(2*len(left_slopes)))
        left_y_avg = int(left_y/(2*len(left_slopes)))
        left_y1 = int(ydimension/2) # Half the image height
        left_x1 = int((left_y1-left_y_avg)/left_slopes_avg + left_x_avg) #x=(y-c)/m
        left_y2 = int(ydimension)
        left_x2 = int((left_y2 - left_y_avg)/left_slopes_avg + left_x_avg)
        
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color=(255,0,0), thickness=10)
        
    if len(right_slopes)!=0:
        right_slopes_avg = np.sum(right_slopes)/len(right_slopes)
        right_x_avg = int(right_x/(2*len(right_slopes)))
        right_y_avg = int(right_y/(2*len(right_slopes)))
        right_y1 = int(ydimension/2) # Half the image height
        right_x1 = int((right_y1-right_y_avg)/right_slopes_avg + right_x_avg) #x=(y-c)/m
        right_y2 = int(ydimension)
        right_x2 = int((right_y2 - right_y_avg)/right_slopes_avg + right_x_avg)
        
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color=(255,0,0), thickness=10)        
    
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
#reading in an image
FirstImage = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#converting to Grayscale image
GrayScaleFirstImage = grayscale(FirstImage)

#apply Gaussian Bluring to supress noise and gradients in the image. Adjust kernel size.
kernel_size = 5
GaussianBlurFirstImage = gaussian_blur(GrayScaleFirstImage, kernel_size)

#apply Canny Edge Detection on a Gaussian smoother image
low_threshold = 50
high_threshold = 100
CannyFirstImage = canny(GaussianBlurFirstImage, low_threshold, high_threshold)
plt.imshow(CannyFirstImage,cmap = 'Greys_r')

#Defining a Polygon formed from vertices.
ImageDimensions = FirstImage.shape
Dimension_y = ImageDimensions[0]
Dimension_x = ImageDimensions[1]

#vertices = np.array([[(Dimension_x * 0.1, Dimension_y * 1.0), (Dimension_x * 0.1, Dimension_y * 0.4), (Dimension_x * 0.6, Dimension_y * 0.4), (Dimension_x * 1.0, Dimension_y * 1.0)]] ,dtype=np.int32)
vertices = np.array([[(0, 500), (400, 340), (600, 340), (850, 500)]] ,dtype=np.int32)

# Return an image with region of interest
RegionFirstImage = region_of_interest(CannyFirstImage, vertices)

#Hough transform to draw lines from points generated previously in Canny Edge Detection
rho = 1
theta = np.pi/180
threshold = 1
min_line_len = 5
max_line_gap = 20
HoughLines = hough_lines(RegionFirstImage, rho, theta, threshold, min_line_len, max_line_gap)

#Overlay Hough lines on original image
ResultFirstImage = weighted_img(HoughLines, FirstImage, α=0.8, β=1., γ=0.)
plt.imshow(ResultFirstImage)
# then save them to the test_images_output directory.
mpimg.imsave("test_images_output/whiteCarLaneSwitch.png", ResultFirstImage, cmap='gray')