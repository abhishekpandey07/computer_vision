''' explore_video.py| created 21 August 2017.
    Author: Abhishek Pandey
    
    The script can be used to explore a video by passing it's address to
    the VideoCapture('file') object creation. Later, a frame can be saved to 
    the destination in string filename- by pressing 'c', provided the directory
    already exists.
    To exit the video press ESC.

    USAGE:  python explore_video.py '''

import numpy as np
import cv2
import work_frame
import ball_tracker




filename = 'dataset/alain_background'
filename_b = 'dataset/balls/ball'
count = 0
ext = '.tiff'
frametoquit = 10

cap =  cv2.VideoCapture('dataset/Alain-pink.mkv')
background = cv2.imread('dataset/alain_background_final.tiff')
print background
_,_,points = work_frame.findAndDrawContours(background.copy())
background = work_frame.changePerspective(points,background)

if not(cap.isOpened()):
    cap.open()
global ball
ball = 99999

def grabBalls(img,circles):
    balls = []
    for [(x,y),r] in circles:
        x = x - 15
        y = y - 10
        w = 25
        h = 20
        balls.append(img[y:y+h,x:x+w])
    return balls
while True:
    ret, frame = cap.read()
    balls, circles = work_frame.process_frame(frame.copy(),background,points)
    save = balls.copy()
    if ret:
         text = 'captures = ' + str(count)
         text2 = 'balls = '+ str(ball)
         cv2.putText(balls, text,(5,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,200,0),2)
         cv2.putText(balls, text2,(250,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,200,0),2)
         cv2.imshow('Break',balls)
    else:
        frametoquit =  frametoquit - 1
        
    k = cv2.waitKey(1) & 0xFF
    if  k==27 or frametoquit == 0:
        break
    elif k == ord('c'):
        count = count + 1
        file  =  filename + '-'+ str(count) + ext
        cv2.imwrite(file, save)  
    elif k == ord('b'):
        balls = grabBalls(save,circles)
        for b in balls:
            ball = ball + 1
            file  =  filename_b + '-'+ str(ball) + ext
            cv2.imwrite(file, b)
    
        
        
cap.release()
cv2.destroyAllWindows()

        
