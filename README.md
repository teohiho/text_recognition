# opencv-text-detection

This is a derivative of [pyimagesearch.com opencv-text-detection](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) and the [opencv text detection c++ example](https://docs.opencv.org/master/db/da4/samples_2dnn_2text_detection_8cpp-example.html)

This code began as an attempt to rotate the rectangles found by EAST.  The pyimagesearch code doesn't rotate the rectangles due to a limitation in the opencv Python bindings -- specifically, there isn't a NMSBoxes call for rotated rectangles (this turns out to be a bigger problem)

[EAST](https://arxiv.org/abs/1704.03155) Efficient and Accurate Scene Text detection pipeline.  Adrian's post does a great job explaining EAST.  In summary, EAST detects text in an image (or video) and provides geometry and confidence scores for each block of text it detects.  Its worth noting that:

* The geometry of the rectangles is given as offsetX, offsetY, top, right, bottom, left, theta.  
* Top is the distance from the offset point to the top of the rectangle, right is the distance from the offset point to the right edge of the rectangle and so on.  The offset point is most likey **not** the center of the rectangle. 
* The rectangle is rotated around **the offset point** by theta radians.

While the EAST paper is pretty clear about determining the positioning and size of the rectangle, its not very clear about the rotation point for the rectangle.  I used the offset point as it appeared to provide the best visual results.

## Modifications & Challenges
In the PyImageSearch example code, Non Maximal Suppression (NMS) is performed on the results provided by EAST on the unrotated rectangles.  The unrotated rectangles returned by NMS are then drawn on the original image.

Initially, I modified the code to rotate the rectangles returned by NMS and then drawing them on the original image.

![Unrotated](images/out/lebron_james_unrotated.jpg)


