import math
import cv2
import torch

# set parameters for superpixel segmentation
num_superpixels = 400  # desired number of superpixels
num_iterations = 4     # number of pixel level iterations. The higher, the better quality
prior = 2              # for shape smoothing term. must be [0, 5]
num_levels = 4
num_histogram_bins = 5 # number of histogram bins


def Integrated_scoremap_generation(I, S_p, P, M, alpha=1., beta=1., gamma=-0.5, phi=4.):
    # superpixel선언
    height, width, channels = I.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)

    # S_sp 생성
    seeds.iterate(I, num_iterations)
    L_sp = seeds.getLabels()

    ## set scores of these pixels the same as the score of matched point (xm, ym)
    for p in M:
        L_sp[p[0][0], p[0][1]] = p[1]
    S_sp = L_sp

    # S_p 생성
    temp = torch.zeros_like(I)

    ## set Sp(x, y) = S(x, y) for (x, y) in the scope of pp˜ which contains matched points, otherwise Sp(x, y) = 0
    for pp in P:
        pp=pp.flatten()
        temp[pp[0]:pp[2],pp[1]:pp[3]]=S_p[pp[0]:pp[2],pp[1]:pp[3]]
    S_p = temp

    return 1/(1+math.exp(-phi(alpha*S_sp+beta*S_p+gamma)))