# -*- coding: UTF-8 -*-
'''
@Project : Fusion Image evaluation 
@File    : Qabf.py
@Author  : Zhiheng Liu
@Email   : visitorindark@gmail.com
@Date    : 2024/07/29 22:39
'''
import numpy as np
import cv2
from scipy.ndimage import convolve

def Qabf(imgA_path, imgB_path, imgF_path):
    """
    compute the QABF metric for the given images
    :param imgA_path: path to the first image
    :param imgB_path: path to the second image
    :param imgF_path: path to the fused image
    :return: QABF metric value

    Reference:
    DOI:10.1049/el:20000267
    https://www.mathworks.com/matlabcentral/fileexchange/18213-objective-image-fusion-performance-measure
    """
    # Model parameters
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    epsilon = 1e-10

    # Sobel Operator
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Load images
    pA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    pB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    pF = cv2.imread(imgF_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    # Compute gradients
    SAx = convolve(pA, h3)
    SAy = convolve(pA, h1)
    gA = np.sqrt(SAx ** 2 + SAy ** 2)
    aA = np.arctan2(SAy, SAx)

    SBx = convolve(pB, h3)
    SBy = convolve(pB, h1)
    gB = np.sqrt(SBx ** 2 + SBy ** 2)
    aB = np.arctan2(SBy, SBx)

    SFx = convolve(pF, h3)
    SFy = convolve(pF, h1)
    gF = np.sqrt(SFx ** 2 + SFy ** 2)
    aF = np.arctan2(SFy, SFx)

    # Relative strength and orientation values
    GAF = np.minimum(gF / (gA + epsilon), gA / (gF + epsilon))
    AAF = 1 - np.abs(aA - aF) / (np.pi / 2)

    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    GBF = np.minimum(gF / (gB + epsilon), gB / (gF + epsilon))
    ABF = 1 - np.abs(aB - aF) / (np.pi / 2)

    QgBF = Tg / (1 + np.exp(kg * (GBF - Dg)))
    QaBF = Ta / (1 + np.exp(ka * (ABF - Da)))
    QBF = QgBF * QaBF

    # Compute QABF
    deno = np.sum(gA + gB + epsilon)
    nume = np.sum(QAF * gA + QBF * gB)
    output = nume / deno

    return output

if __name__ == '__main__':
    # Example usage
    imgA_path = 'Resources/Patient_12_MR_cropped_000_real_A.png'
    imgB_path = 'Resources/Patient_12_MR_cropped_000_real_B.png'
    imgF_path = 'Resources/Patient_12_MR_cropped_000_real_F.png'
    output = Qabf(imgA_path, imgB_path, imgF_path)
    print("QABF:", output)
