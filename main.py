import cv2
import time
vid = cv2.VideoCapture(0) 
img1 = cv2.imread('faces/elonmusk.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, frame = vid.read() 
if ret:
    cv2.imshow("GeeksForGeeks", frame)
    cv2.imwrite("GeeksForGeeks.png", frame) 

    # If keyboard interrupt occurs, destroy image  
    # window 
    cv2.waitKey(0) 
    cv2.destroyWindow("GeeksForGeeks") 
    time.sleep

