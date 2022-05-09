"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np
import torch

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        tuple of floats (dice score of class 1, dice score of class 2, dice score of whole object)
    """

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    try:
        a, b = a.numpy(), b.numpy()
    except:
        a = a.numpy()

    

    intersection = np.sum((a > 0) * (b > 0))
    sum_volumes = np.sum(a > 0) + np.sum(b > 0)
    dc = 2 * float(intersection) / float(sum_volumes)

    intersection1 = np.sum((a == 1) * (b == 1))
    sum_volumes1 = np.sum(a == 1) + np.sum(b == 1)
    dc1 = 2 * float(intersection1) / float(sum_volumes1)

    intersection2 = np.sum((a == 2) * (b == 2))
    sum_volumes2 = np.sum(a == 2) + np.sum(b == 2)
    dc2 = 2 * float(intersection2) / float(sum_volumes2)

    return dc1, dc2, dc


def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    try:
        a, b = a.numpy(), b.numpy()
    except:
        a = a.numpy()

    intersection = np.sum((a > 0) * (b > 0))
    union = np.sum(a > 0) + np.sum(b > 0) - intersection

    return float(intersection) / float(union)