import cv2
import numpy as np


#########################################################################################


def imageProcessing(image) :
    result = imageCopy(image)
    return result


#########################################################################################


def processSingleImage(imagePath, outputPath) :
    # Load Image
    image = imageRead(imagePath)
    
    # Show Loaded Image
    imageShow("Opened Image", image)
    
    # Copy Loaded Image
    result = imageProcessing(image)
    
    # Show Copied Image
    imageShow("Result Image (Copied Image)", result)
    
    # Save Copied Image
    imageWrite(outputPath, result)

#########################################################################################


def processingSingleVideo(videoPath, outputPath) :
    Video(videoPath, outputPath)
    

def imageRead(imagePath, flag=cv2.IMREAD_UNCHANGED) :
    # If No Korea in Path
    # image = cv2.imread(imagePath, flag)
    
    # If Korean in Path
    imageArray = np.fromfile(imagePath, np.uint8)
    image = cv2.imdecode(imageArray, flag)
    
    # Deal with Error
    if image is not None :
        print("Image Opened")
        return image
    else :
        print("Image Not Opened")
        print("Program Abort")
        exit()


def imageShow(imageName, image, flag=cv2.WINDOW_AUTOSIZE) :
    # Show Image using GUI
    cv2.namedWindow(imageName, flag)
    cv2.imshow(imageName, image)
    cv2.waitKey()


def imageWrite(imageName, image) :
    # Save Image
    return cv2.imwrite(imageName, image)


def imageCopy(image) :
    # Copy Image (Call-By-Memory)
    return np.copy(image)


def Video(videoPath, savePath=None) :
    # Load Video
    cap = cv2.VideoCapture(videoPath)
    
    if cap.isOpened() :
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    
    if savePath is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(savePath, fourcc, fps, (width, height), True)
    
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    
    while cap.isOpened() :
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = imageProcessing(frame)
            
            if out is not None:
                # Write frame-by-frame
                out.write(output)
            
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q') :
            break
    
    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()

    cv2.destroyAllWindows()


def imageParameters(imageName, image) :
    height, width = image.shape[0], image.shape[1]
    print("{}.shape is {}".format(imageName, image.shape))
    print("{}.shape[0] is height: {}".format(imageName, height))
    print("{}.shape[1] is width: {}".format(imageName, width))
    if len(image.shape) == 2 :
        print("This is grayscale image.")
        print("{}.shape[2] is Not exist, So channel is 1".format(imageName))
    else:
        print("This is not grayscale image.")
        print("{}.shape[2] is channels: {}".format(imageName, image.shape[2]))
    print("{}.dtype is {}".format(imageName, image.dtype))
    return height, width


def getPixel(image, x, y, c=None) :
    return image[y, x, c]


def setPixel(image, x, y, value, c=None) :
    image[y, x, c] = value
    return image


def CutRectROI(image, x1, y1, x2, y2) :
    return image[y1:y2, x1:x2]


def PasteRectROI(image, x1, y1, dst) :
    y2, x2 = image.shape[:2]
    dst[y1:y1+y2, x1:x1+x2] = image
    return dst
    

def makeBlackImage(image, color=False) :
    height, width = image.shape[0], image.shape[1]
    if color is True :
        return np.zeros((height, width, 3), np.uint8)
    else:
        if len(image.shape) == 2 :
            return np.zeros((height, width), np.uint8)
        else:
            return np.zeros((height, width, 3), np.uint8)


def fillPolyROI(image, points) :
    if len(image.shape) == 2 :
        channels = 1
    else:
        channels = image.shape[2]
    mask = makeBlackImage(image)
    ignoreMaskColor = (255,)*channels
    cv2.fillPoly(mask, points, ignoreMaskColor)
    return mask


def polyROI(image, points) :
    mask = fillPolyROI(image, points)
    return cv2.bitwise_and(image, mask)


def convertColor(image, flag=cv2.COLOR_BGR2GRAY) :
    return cv2.cvtColor(image, flag)


def splitImage(image) :
    return cv2.split(image)


def mergeImage(channel1, channel2, channel3) :
    return cv2.merge((channel1, channel2, channel3))


def rangeColor(image, lower, upper) :
    result = imageCopy(image)
    return cv2.inRange(result, lower, upper)


def splitColor(image, lower, upper) :
    result = imageCopy(image)
    mask = rangeColor(result, lower, upper)
    return cv2.bitwise_and(result, result, mask=mask)


def drawLine(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    return cv2.line(result, point1, point2, color, thickness, lineType)


def drawRect(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    return cv2.rectangle(result, point1, point2, color, thickness, lineType)


def drawCircle(image, center, radius, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    return cv2.circle(result, center, radius, color, thickness, lineType)


def drawEllipse(image, center, axis, angle, startAngle, endAngle, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    return cv2.ellipse(result, center, axis, angle, startAngle, endAngle, color, thickness, lineType)


def drawPolygon(image, pts, isClosed, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(result, [pts], isClosed, color, thickness, lineType)
    

def drawText(image, text, point=(10, 10), font=cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=(255,255,255), thickness=3, lineType=cv2.LINE_AA) :
    result = imageCopy(image)
    return cv2.putText(result, text, point, font, fontScale, color, thickness, lineType)
    

def addImage(image1, image2) :
    return cv2.add(image1, image2)


def addWeightedImage(image1, w1, imagw2, w2=None) :
    if w2 is None:
        return cv2.addWeighted(image1, float(w1) * 0.01, imagw2, float(100 - w1) * 0.01, 0)
    else:
        return cv2.addWeighted(image1, w1 * 0.01, imagw2, w2 * 0.01, 0)


def imageThreshold(image, thresh=128, maxval=255, type=cv2.THRESH_BINARY) :
    _, res = cv2.threshold(image, thresh=thresh, maxval=maxval, type=type)
    return res


def imageBlur(image, ksize) :
    size = ((ksize+1)*2 - 1, (ksize+1)*2 - 1)
    return cv2.blur(image, size)


def imageGaussianBlur(image, ksize, sigmaX, sigmaY) :
    size = ((ksize+1)*2 - 1, (ksize+1)*2 - 1)
    return cv2.GaussianBlur(image, ksize=size, sigmaX=sigmaX, sigmaY=sigmaY)


def imageMedianBlur(image, size) :
    ksize = (size+1)*2 - 1
    return cv2.medianBlur(image, ksize)


def imageBilateralFilter(image, size, sigmaColor, sigmaSpace) :
    d = (size+1)*2 - 1
    return cv2.bilateralFilter(image, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


def imageFiltering(image, kernel, ddepth=-1) :
    return cv2.filter2D(image, ddepth, kernel)
    

def imageEdgePrewitt(image) :
    kernelX = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], np.float32)
    kernelY = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], np.float32)
    imageDeltaX = imageFiltering(image, kernelX)
    imageDeltaY = imageFiltering(image, kernelY)
    return imageDeltaX + imageDeltaY


def imageEdgeSobel(image) :
    imageDeltaX = cv2.Sobel(image, -1, 1, 0, ksize=3)
    imageDeltaY = cv2.Sobel(image, -1, 0, 1, ksize=3)
    return imageDeltaX + imageDeltaY


def imageEdgeScharr(image) :
    imageDeltaX = cv2.Scharr(image, -1, 1, 0)
    imageDeltaY = cv2.Scharr(image, -1, 0, 1)
    return imageDeltaX + imageDeltaY


def imageEdgeLaplacianCV(image) :
    return cv2.Laplacian(image, -1)


def imageEdgeLaplacianFilter2D(image) :
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], np.float32)
    return imageFiltering(image, kernel)


def imageEdgeLoG(image) :
    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]], np.float32)
    return imageFiltering(image, kernel)


def cannyEdge(image, threshold1=100, threshold2=200) :
    return cv2.Canny(image, threshold1, threshold2)


def autoCanny(image, sigma=0.33) :
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Compute the Median of the Single channel Pixel Intensities
    v = np.median(image)
    
    # Apply Automatic Canny Edge Detection Using the Computed Median
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
	
    # Return the Edged Image
    return edged


def computeHist(image, mask=None) :
    bins = np.arange(256).reshape(256,1)
    
    if len(image.shape) == 2 :
        h = np.zeros((300,256,1))
        histItem = cv2.calcHist([image], [0], None, [256], [0,255])
        cv2.normalize(histItem, histItem, 0, 255, cv2.NORM_MINMAX) 
        hist = np.int32(np.around(histItem))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, 255)
        
    elif len(image.shape) == 3 :
        h = np.zeros((300,256,3))
        color = [(255,0,0), (0,255,0), (0,0,255)] 
        for ch, col in enumerate(color) :
            histItem = cv2.calcHist([image], [ch], None, [256], [0,255]) 
            cv2.normalize(histItem, histItem, 0, 255, cv2.NORM_MINMAX) 
            hist = np.int32(np.around(histItem)) 
            pts = np.column_stack((bins, hist)) 
            cv2.polylines(h, [pts], False, col)
    
    return np.flipud(h)


def histogramEqualize(image) :
    if len(image.shape) == 2 :
        return cv2.equalizeHist(image)
    else :
        ch1, ch2, ch3 = splitImage(image)
        ch1Eq = histogramEqualize(ch1)
        ch2Eq = histogramEqualize(ch2)
        ch3Eq = histogramEqualize(ch3)
        return mergeImage(ch1Eq, ch2Eq, ch3Eq)


def imageDilation(image, kernel, iterations) :
    return cv2.dilate(image, kernel=kernel, iterations=iterations)


def imageErosion(image, kernel, iterations) :
    return cv2.erode(image, kernel=kernel, iterations=iterations)


def imageMorphologicalGradient(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8)
    dilation = imageDilation(image, kernel, iterations)
    erosion = imageErosion(image, kernel, iterations)
    return dilation-erosion


def imageOpening(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8)
    erosion = imageErosion(image, kernel, iterations)
    return imageDilation(erosion, kernel, iterations)
    

def imageClosing(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8)
    dilation = imageDilation(image, kernel, iterations)
    return imageErosion(dilation, kernel, iterations)


def imageMorphologyKernel(flag=cv2.MORPH_RECT, size=5) :
    return cv2.getStructuringElement(flag, (size, size))
    
    
def imageMorphologyEx(image, op, kernel, iterations=1) :
    return cv2.morphologyEx(image, op=op, kernel=kernel, iterations=iterations)


def imageResize(image, dsize=None, fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR) :
    if dsize is None and fx == 0.0 and fy == 0.0:
        fx = 1.0
        fy = 1.0
    return cv2.resize(image, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)


def imageTranslation(image, size=None, dx=0.0, dy=0.0, flags=cv2.INTER_LINEAR) :
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    if size is None:
        rows, cols = image.shape[:2]
        size = (cols, rows)
    return cv2.warpAffine(image, M, size, flags=flags)


def imageRotation(image, center=None, angle=0.0, scale=1.0, size=None, flags=cv2.INTER_LINEAR) :
    if center is None:
        rows, cols = image.shape[:2]
        center = (cols/2, rows/2)
    if size is None:
        rows, cols = image.shape[:2]
        size = (cols, rows)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, size, flags=flags)


def imageAffineTransformation(image, image_pts, dst_pts, size=None, flags=cv2.INTER_LINEAR) :
    if size is None:
        rows, cols = image.shape[:2]
        size = (cols, rows)
    M = cv2.getAffineTransform(image_pts, dst_pts)
    return cv2.warpAffine(image, M, dsize=size, flags=flags)


def imagePerspectiveTransformation(image, image_pts, dst_pts, size=None, flags=cv2.INTER_LANCZOS4) :
    if size is None:
        rows, cols = image.shape[:2]
        size = (cols, rows)
    M = cv2.getPerspectiveTransform(image_pts, dst_pts)
    return cv2.warpPerspective(image, M, dsize=size, flags=flags)


def houghLines(image, rho=1, theta=np.pi/180, threshold=100) :
    return cv2.HoughLines(image, rho, theta, threshold)


def drawHoughLines(image, lines) :
    result = imageCopy(image)
    if len(image.shape) == 2 :
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(lines)) :
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return result


def houghLinesP(image, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=100) :
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


def drawHoughLinesP(image, lines) :
    result = imageCopy(image)
    if len(image.shape) == 2 :
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(lines)) :
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return result


def splitTwoSideLines(lines, slope_threshold = (5. * np.pi / 180.)) :
    lefts = []
    rights = []
    for line in lines:
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        if (x2-x1) == 0:
            continue
        slope = (float)(y2-y1)/(float)(x2-x1)
        if abs(slope) < slope_threshold:
            continue
        if slope <= 0:
            lefts.append([slope, x1, y1, x2, y2])
        else:
            rights.append([slope, x1, y1, x2, y2])
    return lefts, rights


def splitOneSideLines(lines, slope_threshold = (5. * np.pi / 180.)) :
    arranged_lines = []
    for line in lines:
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        if (x2-x1) == 0:
            continue
        slope = (float)(y2-y1)/(float)(x2-x1)
        if abs(slope) < slope_threshold:
            continue
        arranged_lines.append([slope, x1, y1, x2, y2])
    return arranged_lines


def medianPoint(x) :
    if len(x) == 0:
        return None
    else:
        xx = sorted(x)
        return xx[(int)(len(xx)/2)]


def interpolate(x1, y1, x2, y2, y) :
    return int(float(y - y1) * float(x2-x1) / float(y2-y1) + x1)


def lineFittingOneSide(image, lines, color = (0,0,255), thickness = 3, slope_threshold = (5. * np.pi / 180.)) :
    result = imageCopy(image)
    height = image.shape[0]
    arrangedLines = splitOneSideLines(lines, slope_threshold)
    medianLine = medianPoint(arrangedLines)
    min_y = int(height*0.6)
    max_y = height
    min_x = interpolate(medianLine[1], medianLine[2], medianLine[3], medianLine[4], min_y)
    max_x = interpolate(medianLine[1], medianLine[2], medianLine[3], medianLine[4], max_y)
    cv2.line(result, (min_x, min_y), (max_x, max_y), color, thickness)
    return result


def lineFitting(image, lines, color = (0,0,255), thickness = 3, slope_threshold = (5. * np.pi / 180.)) :
    result = imageCopy(image)
    height = image.shape[0]
    lefts, rights = splitTwoSideLines(lines, slope_threshold)
    left = medianPoint(lefts)
    right = medianPoint(rights)
    min_y = int(height*0.6)
    max_y = height
    if left is not None:
        min_x_left = interpolate(left[1], left[2], left[3], left[4], min_y)
        max_x_left = interpolate(left[1], left[2], left[3], left[4], max_y)
        cv2.line(result, (min_x_left, min_y), (max_x_left, max_y), color, thickness)
    if right is not None:
        min_x_right = interpolate(right[1], right[2], right[3], right[4], min_y)
        max_x_right = interpolate(right[1], right[2], right[3], right[4], max_y)
        cv2.line(result, (min_x_right, min_y), (max_x_right, max_y), color, thickness)
    return result


def houghCircles(image, method=cv2.HOUGH_GRADIENT, dp = 1, minDist = 10, canny = 50, threshold = 30, minRadius = 0, maxRadius = 0) :
    circles = cv2.HoughCircles(image, method, dp, minDist, param1=canny, param2=threshold, minRadius=minRadius, maxRadius=maxRadius)
    return circles


def drawHoughCircles(image, circles) :
    result = imageCopy(image)
    if circles is None:
        return result
    for i in circles[0,:]:
        cx = int(i[0])
        cy = int(i[1])
        rr = int(i[2])
        cv2.circle(result, (cx,cy), rr, (0, 255, 0), 2)
        cv2.circle(result, (cx,cy), 2, (0, 0, 255), -1)
    return result
