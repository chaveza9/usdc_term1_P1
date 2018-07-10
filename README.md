
# Project: **Finding Lane Lines on the Road**
***

## Objective:
---
The idea of the project is generate a robust lane lines identifier on a video using computer vision techniques. For this purpose, a python pipeline was programmed using the OpenCV library to extract the lane lines and then draw two straight lines using the extracted lines. These bigger lines represent the lanes on the road. Thus, for this purpose the following especific objectives were defined :

- Read the image
- Generate a color mask to filter the yellow and white lane lines
- Compute the grayscape and eliminate image noise of the masked image
- Detect image edges using Canny edge dectection algorithms
- Compute lines using Hough Transform
- Extrapolate/Average the lane lines
- Apply the algorithm on a video 

The pipeline was tested with a set of road images, and then with a set of videos. The outputs were then stored and then shown inside the "test_videos_output" folder. 

The chosen approach for each objective is explained below.



### 1. Read RGB Image
***

The first step towards the localization of the lane lines is to read each frame of the video. For this purpose we start with a set of color images represented in the red, green, and blue space (RGB). This means that the image $I\in R^{m\times n}$ is defined as a three channel matrix, where m and n represent the image size in x and y direction respectively. Each value of the matrix represent a byte value (0-255) that contains the pixel intensity of each color. 

In order to read the set of images, two different functions were created in order to read and display a set of images.


```python
    
    def read_image_set(fldr):
        """
        'fldr' is a string containing the test images folder
        the return is a list of multidimensional arrays
        """   
        return [mpimg.imread(fldr+img) for img in os.listdir(fldr)]
    def show_image_list(imgList,title=None):
        """ 
        show_image_list(imgList):
        Generate image subplot for visualization of 'imgList' as a grid of subplots
        --------------------------------------------------------------------------------------------
        INPUT:
            imgList: A list of multidimensional arrays containing the images
            title : String containing title of superplot
        OUTPUT:
            none
        ============================================================================================
        """
        gridSize = [2,int(len(imgList)/2)] 
        fig = plt.figure()
        for indx, img in enumerate(imgList):
            fig.add_subplot(gridSize[0],gridSize[1],indx+1)
            if len(img.shape)==3:
                imgplot = plt.imshow(img)
            else:
                imgplot = plt.imshow(img,cmap='gray')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.001)
        fig.suptitle(title)  
        plt.rcParams["figure.figsize"] = [16,9]
        plt.show()
        return fig
    
```

<p align="center">
 <img src="/test_images_output/RGB.png" alt="RGB Images" />
</p>


### 2. Color Mask

***

The idea of using a color mask is to be able to filter out the unwanted objects that can alter the extraction of the desired features. In this case, the objective is to extract just the lane lines, which by convention, are either yellow or white. However, there exist different intensities of the same color that depend on the light conditions, thus afecting the robustness of the color mask algorithm. Therefore, an image mask algorithm is more robust when the *luma* ,or light intensity, is separated from the the color information on the image space. A color space alternative that takes the luma as a separate parameter is the hue, saturation, lightness (HLS) space.

The HLS space is represented with a cylindrical geometry in mind, where hue is a parameter in the angular dimension, saturation is in the radial direction, and lightness is in the lenght dimention. This configurations allows the definition of an HLS color range that takes into consideration different light conditions. Therefore, the color mask that has been defined into the pipeline is based on an HLS color space. 

The first step towards the computation of the color mask is to transform the RGB image into the HLS color space. For this purpose we use the `cv2.cvtColor(img, cv2.COLOR_RGB2HLS)` function. The transformation of the RGB test images into the HLS space is shown below.

<p align="center">
 <img src="/test_images_output/HLS.png" alt="HLS Images" />
</p>

Once the image has been converted into the HLS space, a HSL color range can be defined as a filter in the HLS space. The chosen ranges for the yellow and white lines are defined respectively in the tables below

<p align="center">
    White
</p>

   | Hue      | Light     | Saturation|
---|:--------:| :--------:| :--------:|
Min| 0        | 220       | 0         |
Max| 255      | 255       | 255       |

<p align="center">
    Yellow
</p>

   | Hue      | Light     | Saturation|
---|:--------:| :--------:| :--------:|
Min| 0        | 0         | 150       |
Max| 80       | 255       | 255       |

Thus, the color mask is defined as follows:

```python
def mask_color_lanes(img):
    """
    mask_color_lanes(img)
    Applies a color based mask to an rgb image by converting it to an HSV color
    image
    ---------------------------------------------------------------------------
    INPUT:
        img: RGB color image
    OUTPUT:
        maskedImg: Single Channel Masked image
    ===========================================================================
    """
    hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Define Upper and Lower Thresholds for white and yellow lanes
    white = {'low':np.uint8([  0, 220,   0]),'up':np.uint8([255, 255, 255])}
    yellow = {'low':np.uint8([0, 0, 150]),'up':np.uint8([80, 255, 255])}
    
    # Apply a mask by filtering colors that are in range
    white_mask = cv2.inRange(hlsImg, white['low'], white['up'])  
    yellow_mask = cv2.inRange(hlsImg,  yellow['low'], yellow['up'])
    
    # Combine masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Apply mask to image
    maskedImg = cv2.bitwise_and(img, img, mask=mask)
    
    return maskedImg
```
<p align="center">
 <img src="/test_images_output/ColorMasked.png" alt="HLS Masked Images" />
</p>


### 3. Grayscale and Noise Filtering

***

Once the image has been filtered for color, the masked image needs to be converted into grayscale, meaning that the image is expressed in a single channel with the dimensions of the image. Each value of the image represent the amount of light that each pixel contains. This transformation is used later in the pipeline so the pixel gradient can be computed. However, before the gradients can be computed, a pre-processing of the image needs to be done. This means that the image color can be smoothed, thus eliminating some of the noise that the image contains. For this purpose, due to simplicity, it is assumed that the image contains gaussian white noise. This process is known as _gaussian bluring_ and is used tipically to reduce detail in images. The distortion of the image depends on kernel size that is used. 

```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

<p align="center">
 <img src="/test_images_output/GaussianBlur.png" alt="Gaussian Blur Images" />
</p>




### 4. Canny Edge Detection

Once the image has been pre-processed, we can start detecting the lane lines by computing the pixel gradients so the edges on an image can be extracted. This algorithm is called _Canny Edge Detection_ ans is named after its creator John F. Canny. In order for the algorithm to be able to define what is an edge, an _Hysteresis Thresholding_ is used. This means that a minimum and a maximum value of intensity.  Source:[Canny Tutorial](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)

An additional step has been adapted from [Adrian Rosebrock](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/). This step computes the hysteresis thresholding using the median of the pixel intensities.

``` Python
def auto_canny(img, sigma=0.33):
    """
    auto_canny(imgList):
    Computes the upper and lower threshold parameters using the median of the 
    pixel intensities for the computation of the Canny transform
    Note: Code adapted from
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    --------------------------------------------------------------------------------------------
    INPUT:
        img: image to apply the canny transform
        sigma : constant parameter for variation and tunning of the threshold percertanges
    OUTPUT:
        edgeImg: single channel image with canny transform
    ============================================================================================
    """
    # Compute the median of the pixel intensities
    m = np.median(img)
    
    # Compute the upper and lower limits for canny parameters using median
    low_threshold = int(max(0, (1.0 - sigma) * m))
    high_threshold = int(max(0, (1.0 + sigma) * m))
    
    return cv2.Canny(img, low_threshold, high_threshold)
```
In order to limit the detection of edges to the region of interest, we can assume that on a car the lane lines are usually in the region of the image. Thus, we can define a region mask as follows:

``` Python
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
```

<p align="center">
 <img src="/test_images_output/Edges.png" alt="Canny Image" />
</p>





### 5. Hough Transform

Once the edges of an image have been detected, one can detect lines by mapping the vectorial space of the edge pixels into the called "Hough Space" to describe a geometrical shape. The Hough line transform uses polar coordinates to describe a line. However, in order to determine whether a set of points belong to a line, a voting scheme is used. The Hough transform algorithm  outputs a set of endpoints that describe a line.

``` Python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
                        minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

```
<p align="center">
 <img src="/test_images_output/HoughLines.png" alt="Hough Lines Image" />
</p>

### 4. Line Averaging/Extrapolation

Once the lines of an image have been detected, we can compute two separate main lines that can be extended along the region of interest of the image using the lines identified with the Hough Transform. However, there exist several parameters that need to be taken into consideration before. At first it is necessary to classify the set of lines into _right_ and _left_ lane lines. For this purpose we can compute the slope of each line using the following equation

$$ m = \dfrac{y_2-y_1}{x_2-x_1} $$


Thus, a line $l_i$ can be classified using its slope to define two different set of lines $ \left( R,L \right) $, that correspond to the right and left lines sets respectively. The classification is computed given the following metric


$$ f(x) =
\left\{
    \begin{array}{ll}
        {l_i \in L}  & \mbox{if } m > 0 \\
        {l_i \in R}  & \mbox{if } m <0 
    \end{array}
\right. $$


This classification is defined as part of the `separate_lines()` function and is defined as follows:


``` Python
def separate_lines(lines,imshape):
    """
    separate_lines(lines)
    Classifies left and right lines  based on slope
    ---------------------------------------------------------------------------
    INPUT:
        lines: line points [[x1,y1,x2,y2]]
    OUTPUT:
        right{}: right line dictionary with the following structure 
            ['slope'] = line slope  (y2-y1)/(x2-x1)
            ['lane']  = [[x1,y1,x2,y2]] 
        left{}:
            ['slope'] = line slope  (y2-y1)/(x2-x1)
            ['lane']  = [[x1,y1,x2,y2]] 
    ===========================================================================
    """
    # Generate a structure of data for left and right lines
    left = {'slope':[],'intercept':[],'lane':np.empty((0,4),dtype=np.int32)}
    
    right = {'slope':[],'intercept':[],'lane':np.empty((0,4),dtype=np.int32)}
    
    # Compute Lines and Separate right and left lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            # Skip vertical line iteration
            if x1==x2 or y1==y2:
                continue       
            m = (y2-y1)/(x2-x1)
            b = y2 - m*x2
            if m < 0 and x2 < imshape[1]/2:
                left['slope'].append(m) 
                left['intercept'].append(b) 
                left['lane'] = np.append(left['lane'],line,axis= 0)
                
            elif m > 0 and x2> imshape[1]/2:
                right['slope'].append(m) 
                right['intercept'].append(b) 
                right['lane'] =np.append(right['lane'],line,axis= 0)
                
    return left,right
```


Once the lines have been classified, the natural approach to generate an average line is compute the mean of the line slope $m$ and its intercept $b$ for each set of lines. However, when the main problem with this approach is when an outlier, that does not belong to the lane lines, has been detected as a line. The effect of the inclusion of outliers when averaging has a repercussion on the stability of the algorithm. Therefore, it is important to be able to detect and reject outliers. For this purpose, we can use the standard deviation of the slopes and the intercepts of each line set assuming that they have a gaussian normal distribution. The outlier rejection code is shown as follows:


```Python
def  outlier_rejection(slope,intercept, lane, std = 1.5):
    """
    outlier_rejection(slope, lane, std)
    Extract outliers based on the standard deviation of the slopes and 
    intercepts
    ---------------------------------------------------------------------------
    INPUT:
        slope: line slope list
        lane : line points [[x1,y1,x2,y2]]
        std : standard deviation threshold
    OUTPUT:
        pts: points [[x,y]] with inliers
    ===========================================================================
    """
    # Compute Inliers index
    indx = (np.abs(slope-np.array(slope).mean())<std*np.array(slope).std()) & \
            (np.abs(intercept-np.array(intercept).mean())<\
             std*np.array(intercept).std())
            
    n = len(np.array(slope)[indx]) # Array Size
    # Extract inliers using index and form points array [[x,y]]
    pts = np.hstack((lane[indx][:,[0,2]].reshape((n*2,1)),\
                    lane[indx][:,[1,3]].reshape((n*2,1))))    
    return pts
```


Source: [Removing Outliers Using Standard Deviation in Python](https://www.kdnuggets.com/2017/02/removing-outliers-standard-deviation-python.html)


After removing the most of the outliers, we can perform an averaging or extrapolation algorithm to obtain the main lines. In this case we have chosen to perform a least squares regression on the endpoints of the lines for each line set. The reason for chosing a regression algorithm over a  weighted mean or a simple line extrapolation is that we can include  cost function that could increase the accuracy of the results. An increased accuracy can be done by reducing the effects of the outliers on the overall solution.  The cost function chosen is the Huber loss, and is defined as:


$$ \rho_{\delta}\left( r \right) =
\left\{
    \begin{array}{ll}
        {r^2 \over 2}  & \mbox{if } r > C \\
        {C \left(r-{\delta \over 2} \right) }  & \mbox{otherwise }  
    \end{array}
\right. $$


where C represents a tunning constant, and r represents the residual and is defined as the difference between the observed and 
predicted values.  Source: [Robust Regression](https://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf), [OpenCV](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#fitline)

For the purpose of using the regression, we can use the OpenCV function `cv2.fitline(points, distType, param, reps, aeps)` as follows



```Python
def fit_line(lines,imshape):
    """
    fit_line(lines)
    Extrapolates lines using linear regression with Hueber Loss distance for 
    the left and right lines
    ---------------------------------------------------------------------------
    INPUT:
        line: line points [[x1,y1,x2,y2]]
    OUTPUT:
        right['coef']: right line coefficients [[m,b]] where m is the slope and
                        b is the line intercept
        left['coef']: right line coefficients [[m,b]] where m is the slope and
                        b is the line intercept
    ===========================================================================
    """
    # Classify left and right lines based on slope
    left,right =  separate_lines(lines,imshape)              
    
    # Discard outliers based on the slope and intercept 
    # standard deviation (Mean + 2*SD)
    left['points']  = outlier_rejection(left['slope'],left['intercept'], \
        left['lane'],1.5)
    right['points'] = outlier_rejection(right['slope'],right['intercept'],\
         right['lane'],1.5)
    
    # Linear Regression on the inliers points using Huber Loss as error 
    # distance DIST_HUBER
    # LEFT:
    l = cv2.fitLine(left['points'], cv2.DIST_HUBER, 0,0.01,0.01)
    m = l[1]/l[0]       # Slope
    b = l[3]-(l[2]*m)   # Intercept
    left['coef'] = [m,b]
    
    #RIGHT:  
    r = cv2.fitLine(right['points'], cv2.DIST_HUBER, 0,0.01,0.01)
    m = r[1]/r[0]       # Slope
    b = r[3]-(r[2]*m)   # Intercept
    right['coef'] = [m,b]

    return left['coef'],right['coef']
```

Putting all together we get

``` Python
def line_computation(img,coef):
    """
    line_computation(img,coef)
    Computes the extreme points of the lane lines based on the image
    coefficients
    ---------------------------------------------------------------------------
    INPUT:
        img: image 
        coef : line coefficients [[m,b]]
    OUTPUT:
        line_pts = Line points [x1,y1,x2,y2]
    ===========================================================================
    """

    # Define Line points [x1,y1,x2,y2]
    line_pts = np.zeros([1,4], dtype=np.int)
    
    # Line equation y = mx+b-->x = (y-b)/m
    line_pts[0][1] = img.shape[0]
    line_pts[0][0] = int((line_pts[0][1]-coef[0][1])/coef[0][0])
    line_pts[0][3] = img.shape[0]*0.62
    line_pts[0][2] = int((line_pts[0][3]-coef[0][1])/coef[0][0])
    
    return line_pts

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once 
    you want to average/extrapolate the line segments you detect to map out the 
    full extent of the lane (going from the result shown in 
    raw-lines-example.mp4 to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left,right = fit_line(lines)
    lane_lines = np.vstack((line_computation(img,left),\
                             line_computation(img,right)))

    for x1,y1,x2,y2 in lane_lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```

<p align="center">
 <img src="/test_images_output/MainLines.png" alt="Main Lines Image" />
</p>

### 5. Applying Algorithm to Video

***
So far, we have defined a pipeline to identify the lane lines on a single image. However, before we can implement this algorithm on a subsequent set of images from a video feed, we need to handle the stability of the line detection. If we apply the pipeline as is, the pipeline can generate two main lines that appear to be moving on each frame. For this purpose, we have need to definea buffer so we can store the line coeffients that we have computed from the regression. Once we have stored the main line coefficients of at least 10 frames, we can average the coefficients stored in the buffer to smooth the vizualization. In order to define the buffer, we need to modify some of the functions previously defined so we can include a global variable. The final functions and pipeline is defined as follows:

``` Python
def line_smoothing (l_coef,r_coef):
    """
    line_smoothing(line_coeff)
    Computes the mean of the previous lane line coefficients to smooth the 
    visualization of the global lane
    ---------------------------------------------------------------------------
    INPUT:
        r_coef: right line coefficients [[m,b]] where m is the slope and
                        b is the line intercept
        l_coef: right line coefficients [[m,b]] where m is the slope and
                        b is the line intercept
    OUTPUT:
        avg_lineCoeff: Average line coefficients
    ===========================================================================
    """
    global global_lanes
    l_coef = np.array(l_coef).T
    r_coef = np.array(r_coef).T
    # Use fifo on the first 5 lines
    if global_lanes.shape[0] >= 10:
        global_lanes = np.delete(global_lanes,0,0)
    # Append line coefficient
    global_lanes = np.append(global_lanes,np.hstack((l_coef,r_coef)),axis=0)
    
    avg_lineCoef = global_lanes.mean(axis=0)    
    return avg_lineCoef[0:2],avg_lineCoef[2:]

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once 
    you want to average/extrapolate the line segments you detect to map out the 
    full extent of the lane (going from the result shown in 
    raw-lines-example.mp4 to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Extrapolate the lines
    left,right = fit_line(lines,img.shape)
    
    # Smooth ouput Coefficients
    l_coef,r_coef = line_smoothing (left,right)
    
    # Compute line extreme points    
    lane_lines = np.vstack((line_computation(img,l_coef),\
                             line_computation(img,r_coef)))
    
    # Draw Lines
    for x1,y1,x2,y2 in lane_lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
                        minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
def process_image(img):
    # Apply Mask on colors
    colorMaskImg = mask_color_lanes(img)
    # Convert Images into Greyscale
    grayImg = grayscale(colorMaskImg)
    # Filter Image using Gaussian Smoothing
    blur_grayImg = gaussian_blur(grayImg,kernel_size=9)
    # Apply an automatic parameter canny detection
    cannyImg = auto_canny(blur_grayImg)
    
    # Apply Mask
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]),(imshape[1]*.45, imshape[0]*0.62), 
                              (imshape[1]*.55, imshape[0]*0.62), (imshape[1],imshape[0])]], 
                                dtype=np.int32)
    maskedImg = region_of_interest(cannyImg,vertices)

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 5 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    hough_linesImg = hough_lines(maskedImg, rho, theta,threshold, min_line_len, max_line_gap)
    result = weighted_img(hough_linesImg,img) 

    return result    
```

## Shortcomings and Potential Improvements

We have developed a pipeline that is capable of generating unbiased results and is robust. However, the main problem with the approach taken is the computation time. The reason for the increased computationtime is the increment of the time complexity when we started using the Hueber Loss regression. A potential fix on this issue is the use of an M estimator such as Msac or Ransac to detect the lines instead of using the Hough transform. The reason is that we can define the cost function as part of the regression.

Additionally, it is important to notice that there exist no distortion correction on the images. Therefore, we can include the camera calibration matrix so we can correct the image distortion.

One of the main limitations of this algorithm, is that we can only generate an extrapolation on straigth, or semi-straight lane lines. We can include curves by increasing the order of the regression to a second order and by detecting parabolas instead of lines. 


