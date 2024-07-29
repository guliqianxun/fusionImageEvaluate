# -*- coding: UTF-8 -*-
'''
@Project : Fusion Image evaluation 
@File    : PiellaMetric.py
@Author  : Zhiheng Liu
@Email   : visitorindark@gmail.com
@Date    : 2024/07/29 16:39
'''
import numpy as np
import cv2

def local_quality_index(img1, img2, window_size=8):
    height, width = img1.shape
    local_qualities = []
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            window1 = img1[i:i+window_size, j:j+window_size]
            window2 = img2[i:i+window_size, j:j+window_size]
            
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)
            std1 = np.std(window1)
            std2 = np.std(window2)
            covariance = np.mean((window1 - mean1) * (window2 - mean2))
            
            numerator = 4 * mean1 * mean2 * covariance
            denominator = (mean1**2 + mean2**2) * (std1**2 + std2**2) + 1e-10
            
            local_quality = numerator / denominator
            local_qualities.append(local_quality)
    
    return np.mean(local_qualities)

def PiellaMetric(imgA, imgB, imgF, window_size=8):
    """
    compute the Piella's metric for the given images
    :param imgA: first image
    :param imgB: second image
    :param imgF: fused image
    :return: Piella's metric value

    Reference:
    A new quality metric for image fusion
    DOI: 10.1109/ICIP.2003.1247209
    """
    # Define the Piella metric parameters
    epsilon = 1e-10 

    height, width = imgA.shape
    saliency_A = np.zeros_like(imgA)
    saliency_B = np.zeros_like(imgB)
    
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            windowA = imgA[i:i+window_size, j:j+window_size]
            windowB = imgB[i:i+window_size, j:j+window_size]
            
            saliency_A[i:i+window_size, j:j+window_size] = np.var(windowA)
            saliency_B[i:i+window_size, j:j+window_size] = np.var(windowB)
    
    total_saliency = saliency_A + saliency_B + 1e-10
    weight_A = saliency_A / total_saliency
    weight_B = saliency_B / total_saliency
    
    Q_AF = local_quality_index(imgA, imgF, window_size)
    Q_BF = local_quality_index(imgB, imgF, window_size)
    
    Q = np.mean(weight_A * Q_AF + weight_B * Q_BF)
    
    return Q

if __name__ == '__main__':
    # Example usage
    imgA_path = 'Resources/Patient_12_MR_cropped_000_real_A.png'
    imgB_path = 'Resources/Patient_12_MR_cropped_000_real_B.png'
    imgF_path = 'Resources/Patient_12_MR_cropped_000_real_F.png'
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    imgF = cv2.imread(imgF_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    Q = PiellaMetric(imgA, imgB, imgF)
    print("Q (Piella's Metric):", Q)
