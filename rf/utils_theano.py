import numpy as np

import theano
import theano.tensor as T
import lasagne

from lasagne.utils import floatX
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer



def prep_image(im,IMAGE_W):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape

    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, int(w*IMAGE_W/h)), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (int(h*IMAGE_W/w), IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

def deprocess(x):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def torch_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    #print(N,M)
    #assert(false)
    
    loss = 1./(M**2) * ((G - A)**2).sum()
    return loss

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def style_loss_relative(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    loss = ((G - A)**2).sum() / (G**2).sum()
    return loss

def build_model_one_scale(IMAGE_W, n_feature_maps, filter_size):
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], n_feature_maps, filter_size, pad=filter_size//2, flip_filters=False)
    return net

def build_model_multiscale(IMAGE_W, n_feature_maps, scales, nonlinearity=lasagne.nonlinearities.rectify, nobias=False):
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))

    if nobias:
        print('no bias in CNN layer')
        multiple_scales = [ConvLayer(net['input'], n_feature_maps,
                                     filter_size, pad=filter_size//2,
                                     flip_filters=False,nonlinearity=nonlinearity,b=None)
                           for filter_size in scales]
    else:
        multiple_scales = [ConvLayer(net['input'], n_feature_maps,
                                     filter_size, pad=filter_size//2,
                                     flip_filters=False,nonlinearity=nonlinearity)
                           for filter_size in scales]
    net['conv1_1'] = ConcatLayer(multiple_scales)
    return net

def build_model_multiscale_gray(IMAGE_W, n_feature_maps, scales, nonlinearity=lasagne.nonlinearities.rectify,nobias=False):
    net = {}
    net['input'] = InputLayer((1, 1, IMAGE_W, IMAGE_W))

    if nobias:
        print('no bias in CNN layer')
        multiple_scales = [ConvLayer(net['input'], n_feature_maps,
                                     filter_size, pad=filter_size//2,
                                     flip_filters=False,nonlinearity=nonlinearity,b=None)
                           for filter_size in scales]
    else:
        multiple_scales = [ConvLayer(net['input'], n_feature_maps,
                                     filter_size, pad=filter_size//2,
                                     flip_filters=False,nonlinearity=nonlinearity)
                           for filter_size in scales]
    
    net['conv1_1'] = ConcatLayer(multiple_scales)
    return net

def build_model_multiscale_gray_torch(IMAGE_W, n_feature_maps, scales, nonlinearity=lasagne.nonlinearities.rectify):
    net = {}
    net['input'] = InputLayer((1, 1, IMAGE_W, IMAGE_W))

    multiple_scales = [ConvLayer(net['input'], n_feature_maps, filter_size,\
                                 W=lasagne.init.Uniform(std=np.sqrt(2.0/(n_feature_maps*filter_size*filter_size))),
                                 b=None,
                                 pad=filter_size//2, flip_filters=False,
                                 nonlinearity=nonlinearity)
                       for filter_size in scales]
    net['conv1_1'] = ConcatLayer(multiple_scales)
    return net
