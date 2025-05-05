import cv2
import numpy as np

from utils import convertColor


#########################################################################################


def nothing(x) :
    pass


#########################################################################################


def main() :
    # Generate Default Black Image
    image = np.zeros((10, 10, 3), np.uint8)
    
    # Create Trackbar for Visualizing the Effect of RGB
    # API Document Link -> https://opencv-python.readthedocs.io/en/latest/doc/05.trackBar/trackBar.html
    cv2.namedWindow("Image", cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("H", "Image", 90, 179, nothing) # 초기값 -> 90 / 최대값 -> 179
    cv2.createTrackbar('L', "Image", 128, 255, nothing) # 초기값 -> 128 / 최대값 -> 255
    cv2.createTrackbar("S", "Image", 128, 255, nothing) # 초기값 -> 128 / 최대값 -> 255

    while True :
        # Show Loaded Image
        cv2.imshow("Image", image)
        
        # ECS Key를 사용하여 프로그램 종료
        if cv2.waitKey(100) & 0xFF == 27 :
            break
        
        h = cv2.getTrackbarPos("H", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        l = cv2.getTrackbarPos('L', "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        s = cv2.getTrackbarPos("S", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        
        # Generate New Image
        image[:] = [h, l, s]
        
        # Color Space Conversion (HLS -> BGR)
        image = convertColor(image, cv2.COLOR_HLS2BGR)
    
    cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Excute Main Function
    main()