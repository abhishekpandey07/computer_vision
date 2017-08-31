import cv2
import numpy as np
from glob import glob

class Snooker:
    def __init__(self):
        self.roi = None

    def max_area_contour(self):
        maximum_area = 0
        select = None
        for cnt in self.contours:
            area =  cv2.contourArea(cnt)
            if area  > maximum_area:
                maximum_area = area
                select = cnt
        return select
                        
    def findROI(self,img):
            lower = np.array([52,0,61])
            higher = np.array([80,255,255])
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower,higher)
            _, self.contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if self.contours is not(None):
                select = self.max_area_contour()
                if select is not None:
                    (x,y,w,h) = cv2.boundingRect(select)
                    self.roi = hsv[y-5:y+h,x:x+w]
                    return  True, img[y-5:y+h,x:x+w]
            return False,None

    def findBalls(self,img):
        reds = []
        colors = np.array((-1,6))
        cueball = None
        lower = np.array([[61,255,185],[0,0,0]])
        higher= np.array([[70,255,193],[20,255,255]])
        mask_blue = cv2.inRange(self.roi,lower[0],higher[0])
        mask_red = cv2.inRange(self.roi, lower[1], higher[1])

        _,contour_blue,_ = cv2.findContours(mask_blue,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _,contour_red,_ = cv2.findContours(mask_red,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        red = np.zeros(self.roi.shape,np.uint8)
        blue = np.zeros(self.roi.shape, np.uint8)

        cv2.drawContours(red,contour_red,-1,(0,0,255),2)
        cv2.drawContours(blue,contour_blue,-1,(255,0,0),2)

        cv2.imshow('red',red)
        cv2.imshow('blue',blue)
        
    

def main():
    extract = False
    calibrate = False
    balls = False
    cap = cv2.VideoCapture('dataset/Ronnie-147.mp4')
    snooker = Snooker()
    while True:
    
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if extract == True:
            ret, roi = snooker.findROI(frame)
            if balls:
                snooker.findBalls(roi)
            cv2.imshow('roi',roi)
        k = cv2.waitKey(5)
        if k == 27:
            break
        if k == ord('e'):
            extract = True
        if k == ord('c'):
            calibrate = not(calibrate)
        if k == ord('b'):
            balls = not(balls)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
