"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(0,Hi):
        for j in range(0,Wi):
            sum = 0
            for ki in range(int(-Hk/2),int(Hk/2)+1):
                for kj in range(int(-Wk/2),int(Wk/2)+1):
                    if i + ki < 0 or j + kj < 0 or i + ki >= Hi or j + kj >= Wi:
                        sum += 0
                    else:
                        sum += image[i + ki][j + kj] * kernel[ki + int(Hk/2)][kj + int(Wk/2)]
            out[i][j] = sum
            
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))  # 根据边缘大小设计更大的图像
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image  # 将原图拷贝到新图中心
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    pad_height = int(Hk / 2)
    pad_width = int(Wk / 2)
    img_padded = zero_pad(image, pad_height, pad_width) 
    for i in range(0,Hi):
        for j in range(0,Wi):
            localX = i + pad_height
            localY = j + pad_width
            imageArea = img_padded[localX - pad_height:localX + pad_height + 1, localY - pad_width:localY + pad_width + 1]
            out[i,j] = np.sum(np.multiply(imageArea, kernel))
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    ### YOUR CODE HERE.reshape
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    
    
    
    pad_height = int(Hk / 2)
    pad_width = int(Wk / 2)
    img_padded = zero_pad(image, pad_height, pad_width) 
    
    kernel_vec = kernel.reshape(Hk*Wk,1)
    img_vec = np.zeros((Hi*Wi,Hk*Wk))
    
    for i in range(Hi):
        for j in range(Wi):
            img_vec[i*Wi + j, :] = img_padded[i:i+Hk,j:j+Wk].reshape(1,Hk*Wk)

    result = np.dot(img_vec, kernel_vec)
    out = result.reshape(Hi,Wi)
    
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g, 0)
    g = np.flip(g, 1)

    out = conv_faster(f,g)
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    ### YOUR CODE HERE
    g = g - g.mean()
    out = cross_correlation(f,g)
    
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    Hg, Wg = g.shape
    Hf, Wf = f.shape
    ### YOUR CODE HERE
    out = np.zeros(f.shape)
    g = (g - np.mean(g)) / np.var(g)
    
    pad_height = int(Hg / 2)
    pad_width = int(Wg / 2)
    f = zero_pad(f, pad_height, pad_width)  # zero padding
    
    for i in range(Hf):
        for j in range(Wf):
            patch = f[i:i+Hg,j:j+Wg]
            patch = (patch - np.mean(patch)) / np.var(patch)
            out[i][j] = np.sum(cross_correlation(patch,g))
    pass
    ### END YOUR CODE

    return out
