import numpy as np
import torch
import scipy.interpolate

def uniform_hist(X):
    ''' 
    Maps data distribution onto uniform histogram

    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))


def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    ''' 
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''
    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
    return matched_image

