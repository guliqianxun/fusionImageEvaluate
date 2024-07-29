# -*- coding: UTF-8 -*-
'''
@Project : Fusion Image evaluation 
@File    : UIQI.py
@Author  : Zhiheng Liu
@Email   : visitorindark@gmail.com
@Date    : 2024/07/29 23:39
'''
import numpy as np
import cv2

def UIQI(img1, img2):
    """
    compute the Universal Image Quality Index (UIQI) for the given images
    :param img1: first image
    :param img2: second image

    Reference:
    Universal image quality index
    DOI: 10.1109/97.995823
    """

    assert img1.shape == img2.shape, "Input images must have the same dimensions"
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    std1 = np.std(img1)
    std2 = np.std(img2)
    covariance = np.mean((img1 - mean1) * (img2 - mean2))
    epilson = 1e-10 #suit for float64

    # Calculate the UIQI components
    numerator = 4 * mean1 * mean2 * covariance
    denominator = (mean1**2 + mean2**2) * (std1**2 + std2**2)
    
    # Calculate UIQI
    Q0 = numerator / (denominator + epilson)  # Adding epsilon to avoid division by zero
    return Q0

if __name__ == '__main__':
    # Example usage
    imgA_path = 'Resources/Patient_12_MR_cropped_000_real_A.png'
    imgB_path = 'Resources/Patient_12_MR_cropped_000_real_B.png'
    imgF_path = 'Resources/Patient_12_MR_cropped_000_real_F.png'
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    imgF = cv2.imread(imgF_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    Q0_A_F = UIQI(imgA, imgF)
    Q0_B_F = UIQI(imgB, imgF)
    print("Q0_A_F:", Q0_A_F)
    print("Q0_B_F:", Q0_B_F)

