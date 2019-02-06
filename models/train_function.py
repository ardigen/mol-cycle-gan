from itertools import chain

import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (Input,
                          BatchNormalization,
                          Lambda)

from .loss import generator_loss, discriminator_loss


def get_train_function(inputs, loss_function, lambda_layer_inputs):
    adam = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-7, decay=0.0)
    train_function = Model(inputs, Lambda(loss_function)(lambda_layer_inputs))
    train_function.compile(adam, 'mse')
    return train_function


def generator_train_function_creator(discriminators_tuple,
                                     generators_tuple,
                                     real_imgs_tuple,
                                     fake_imgs_tuple,
                                     loss_weights_tuple,
                                     use_wgan=False):
    netD_A, netD_B = discriminators_tuple
    netG_A, netG_B = generators_tuple
    real_A, real_B = real_imgs_tuple
    fake_A, fake_B = fake_imgs_tuple
    cycle_loss_weight, id_loss_weight = loss_weights_tuple

    netD_B_predict_fake = netD_B(fake_B)
    rec_A = netG_B(fake_B)

    netD_A_predict_fake = netD_A(fake_A)
    rec_B = netG_A(fake_A)

    lambda_layer_inputs = [netD_B_predict_fake, rec_A, real_A, netD_A_predict_fake, rec_B, real_B, fake_A, fake_B]

    for layer in chain(netG_A.layers, netG_B.layers):
        layer.trainable = True

    for layer in chain(netD_A.layers, netD_B.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netG_loss_partial = lambda x: generator_loss(x,
                                                 cycle_loss_weight=cycle_loss_weight,
                                                 id_loss_weight=id_loss_weight,
                                                 use_wgan=use_wgan)
    netG_train_function = get_train_function(inputs=[real_A, real_B], loss_function=netG_loss_partial,
                                             lambda_layer_inputs=lambda_layer_inputs)
    return netG_train_function


def discriminator_A_train_function_creator(discriminators_tuple,
                                           generators_tuple,
                                           real_imgs_tuple,
                                           input_shape,
                                           use_wgan=False):
    netD_A, netD_B = discriminators_tuple
    netG_A, netG_B = generators_tuple
    real_A, real_B = real_imgs_tuple

    netD_A_predict_real = netD_A(real_A)
    _fake_A = Input(shape=input_shape)
    _netD_A_predict_fake = netD_A(_fake_A)

    for l in netD_A.layers:
        l.trainable = True

    for layer in chain(netG_A.layers, netG_B.layers, netD_B.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netD_loss_partial = lambda x: discriminator_loss(x, use_wgan=use_wgan)
    netD_A_train_function = get_train_function(inputs=[real_A, _fake_A],
                                               loss_function=netD_loss_partial,
                                               lambda_layer_inputs=[netD_A_predict_real,
                                                                    _netD_A_predict_fake])
    return netD_A_train_function


def discriminator_B_train_function_creator(discriminators_tuple,
                                           generators_tuple,
                                           real_imgs_tuple,
                                           input_shape,
                                           use_wgan=False):
    netD_A, netD_B = discriminators_tuple
    netG_A, netG_B = generators_tuple
    real_A, real_B = real_imgs_tuple

    netD_B_predict_real = netD_B(real_B)
    _fake_B = Input(shape=input_shape)
    _netD_B_predict_fake = netD_B(_fake_B)

    for l in netD_B.layers:
        l.trainable = True

    for layer in chain(netG_A.layers, netG_B.layers, netD_A.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netD_loss_partial = lambda x: discriminator_loss(x, use_wgan=use_wgan)
    netD_B_train_function = get_train_function(inputs=[real_B, _fake_B],
                                               loss_function=netD_loss_partial,
                                               lambda_layer_inputs=[netD_B_predict_real,
                                                                    _netD_B_predict_fake])
    return netD_B_train_function


def clip_weights(net, clip_lambda=.1):
    weights = [np.clip(w, -clip_lambda, clip_lambda) for w in net.get_weights()]
    net.set_weights(weights)
