import cv2
import numpy as np


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
    cv2.createTrackbar("R", "Image", 128, 255, nothing) # 초기값 -> 128 / 최대값 -> 255
    cv2.createTrackbar("G", "Image", 128, 255, nothing) # 초기값 -> 128 / 최대값 -> 255
    cv2.createTrackbar("B", "Image", 128, 255, nothing) # 초기값 -> 128 / 최대값 -> 255

    while True :
        # Show Loaded Image
        cv2.imshow("Image", image)
        
        # ECS Key를 사용하여 프로그램 종료
        if cv2.waitKey(100) & 0xFF == 27 : 
            break
        
        r = cv2.getTrackbarPos("R", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        g = cv2.getTrackbarPos("G", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        b = cv2.getTrackbarPos("B", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
        
        # Generate New Image
        image[:] = [b, g, r]

    cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Excute Main Function
    main()