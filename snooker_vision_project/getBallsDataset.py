import cv2
import numpy as np
import common
import ball_tracker
import work_frame
from glob import glob

background_address = 'dataset/background.tiff'

def main():
    files = common.getFiles()
    background = cv2.imread(background_address)
    _,_,points = work_frame.findAndDrawContours(background.copy())
    background = work_frame.changePerspective(points,background)
    back_gray = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
    for f in files:
        img = cv2.imread(f)
        img = work_frame.changePerspective(points,img)
        mask = work_frame.getDilatedMask(img,back_gray)
        circles = ball_tracker.detectBallContour(img,mask)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyWindow('img')
        
if __name__ == '__main__':
    main()
