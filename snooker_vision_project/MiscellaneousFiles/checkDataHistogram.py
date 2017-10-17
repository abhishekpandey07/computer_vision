import cv2
import numpy as np
import common
from glob import glob
import matplotlib.pyplot as plt
import sys
def main(argv):
    global istwoD
    istwoD = argv[1]
    base_directory = 'dataset/balls/balls/'
    labels = ['red','yellow','green','blue','brown',
              'pink','black','cue_ball','not_balls']
    ext = '.tiff'
    for i,l in enumerate(labels):
        
        directory = base_directory + l +  '/*'   + ext
        files = glob(directory)
        img = cv2.imread(files[5])
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        number  =  '52' + str(i+1)
        plt.subplot(int(number))
        if istwoD == 'True':
            hist = common.explore_histogram2d(hsv)
            plt.imshow(hist[0], interpolation = 'nearest')
        else:
            hist = common.explore_histogram(hsv[:,:,0], colour = False)
            colour = ['b','g','r']
            for c,h in enumerate(hist):
                plt.plot(h, color = colour[c] )
                plt.xlim([0,180])
        plt.title(l)

    plt.show()
if  __name__ == '__main__':
    main(sys.argv)
    

    
        
