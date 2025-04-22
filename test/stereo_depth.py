import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


class DepthMap:
    def __init__(self, showImages):
        root = os.getcwd()
        imgLeftPath = os.path.join(root, 'demoImages//motorcycle//ime.png')
        imgRightPath = os.path.join(root, 'demoImages//motorcycle//iml.png')
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show()

    def computeDepthMapBM(self):
        nDispFactor = 6  # adjust this
        stereo = cv.StereoBM.create(numDisparities=16 * nDispFactor, blockSize=21)
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        plt.imshow(disparity, 'gray')
        plt.show()

    def computeDepthMapSGBM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14  # adjust this (14 is good)
        num_disp = 16 * nDispFactor - min_disp
        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=window_size,
                                      P1=8 * 3 * window_size ** 2,
                                      P2=32 * 3 * window_size ** 2,
                                      disp12MaxDiff=1,
                                      uniquenessRatio=15,
                                      speckleWindowsize=0,
                                      speckleRange=2,
                                      preFilterCap=63,
                                      mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
        # Compute disparity map
        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0

        plt.imshow(disparity, 'gray')
        plt.colorbar()
        plt.show()

def demoViewPics():
    dp = DepthMap(True)

def demoComputeBM():
    dp = DepthMap(False)
    dp.computeDepthMapBM()

def demoComputeSGBM():
    dp = DepthMap(False)
    dp.computeDepthMapSGBM()

if __name__ == "__main__":
    demoComputeSGBM()
    # demoComputeBM()
    # demoViewPics()

