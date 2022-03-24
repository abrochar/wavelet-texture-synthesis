# synthesis gray texture, observation pixel value is between [0,1] or zero expectation

import numpy as np
import scipy
import sys
import argparse
import matplotlib.pyplot as plt
import skimage.transform
import scipy.io as sio

import theano
import theano.tensor as T
import lasagne

from lasagne.utils import floatX
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer

from utils import get_gray_mat, prep_image01
from utils_theano import gram_matrix, style_loss, style_loss_relative
from utils_theano import build_model_multiscale_gray

def optimize(texture_name, IMAGE_W, net, n_iter, scales):

    texture0 = get_gray_mat(texture_name,0,IMAGE_W) # (1,size,size)
    pi01 = prep_image01()
    texture = pi01.forward(texture0)
    texture = floatX(texture[np.newaxis]) # (1,1,size,size)
    print('texture',texture.shape)
    
    layers = ['conv1_1']
    layers = {k: net[k] for k in layers}

    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    texture_features = {k: theano.shared(output.eval({input_im_theano: texture}))
                        for k, output in zip(layers.keys(), outputs)}
    
    generated_image = theano.shared(floatX(np.random.uniform(-1, 1, (1, 1, IMAGE_W, IMAGE_W))))
    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
    
    losses = []
    losses_test = []
    factr = 1e7
    losses.append(factr * style_loss(texture_features, gen_features, 'conv1_1'))
    losses_test.append(1 * style_loss_relative(texture_features, gen_features, 'conv1_1'))
    total_loss = sum(losses)
    total_loss_test = sum(losses_test)

    grad = T.grad(total_loss, generated_image)
    f_loss = theano.function([], total_loss)
    f_test_loss = theano.function([], total_loss_test)
    f_grad = theano.function([], grad)

    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 1, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')

    def test_loss(x0):
        x0 = floatX(x0.reshape((1, 1, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_test_loss().astype('float64')

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 1, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')

    texture_init = np.random.uniform(-1, 1, (1, 1, IMAGE_W, IMAGE_W))

    generated_image.set_value(floatX(texture_init))
    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)

    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxiter=n_iter,\
                                 m=20, pgtol=1e-14, factr=10.0, disp=1)
    x0 = generated_image.get_value().astype('float64')
    synthesised = x0[0,:,:,:]
    synthesised = pi01.backward(synthesised)
    synthesised = synthesised[0,:,:] # (H,W)
    
    return synthesised

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--texture', required=True, type=str, help='Path to the reference texture')
    parser.add_argument('-s', '--size', required=True, type=int, help='Size of a synthesised texture in pixels')
    parser.add_argument('-f', '--target-file', default=None, type=str, help='File name of a syntesised texture')
    parser.add_argument('-n', '--n-iter', default=4000, type=int, help='Number of L-BFGS optinisation iterations')
    parser.add_argument('-c', '--n-features', default=128, type=int, help='Number of feature maps per each scale')
    parser.add_argument('--scales', default=None, type=int, nargs='*', help='Sizes of convolutional filters')
    parser.add_argument('-l', '--linear', action='store_true', help='Use linear model (conv layer without non-linearity)')
    args = parser.parse_args()

    texture_name = args.texture

    #global IMAGE_W
    IMAGE_W = args.size

    n_iter = args.n_iter
    n_feature_maps = args.n_features

    # if no scales provided, assume multiscale model
    if args.scales is None:
        scales = [3, 5, 7, 11, 15, 23, 37, 55]
    else:
        scales = args.scales

    print('use scales',scales)
    
    if not args.linear:
        net = build_model_multiscale_gray(IMAGE_W, n_feature_maps, scales, nobias=True) # default has bias
    else:
        net = build_model_multiscale_gray(IMAGE_W, n_feature_maps, scales, nonlinearity=None)

    synthesised = optimize(texture_name, IMAGE_W, net, n_iter, scales)

    if args.target_file is None:
        target_file = texture_name.split('.')[0] +\
                      '_rf_gray_default_nit' + str(n_iter) +\
                      '_size' + str(IMAGE_W) + '.mat'
    else:
        target_file = args.target_file
    sio.savemat(target_file, {'im':synthesised})
    
    return 0

if __name__ == '__main__':
    main()
