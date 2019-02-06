import keras.backend as K
import numpy as np


def criterion_GAN(output, target, use_lsgan=True, use_wesserstein=False):
    assert np.sum(
        [use_lsgan, use_wesserstein]) < 2, "You could use only one of the ['use_lsgan', 'use_wesserstein'] parameters!"

    if use_wesserstein:
        loss = output * target
    elif use_lsgan:
        loss = output - target
        loss = loss ** 2
    else:
        EPS = 1e-6
        loss = (K.log(output + EPS) * target +
                K.log(1 - output + EPS) * (1 - target))

    dims = list(range(1, K.ndim(loss)))
    return K.expand_dims((K.mean(loss, dims)), 0)


def criterion_cycle(reconstructed, real, use_abs=True):
    if use_abs:
        diff = K.abs(reconstructed - real)
    else:
        diff = reconstructed - real
        diff = diff ** 2

    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims((K.mean(diff, dims)), 0)


def compute_similarity_loss(X, real_X, Y, real_Y):
    loss_X = criterion_cycle(X, real_X)
    loss_Y = criterion_cycle(Y, real_Y)
    return loss_X + loss_Y


def generator_loss(G_tensors, cycle_loss_weight=.3, id_loss_weight=.1, use_wgan=False):
    netD_B_predict_fake, rec_A, real_A, netD_A_predict_fake, rec_B, real_B, fake_A, fake_B = G_tensors

    # GAN loss
    if use_wgan:
        G_B_target = -K.ones_like(netD_A_predict_fake)
        G_A_target = -K.ones_like(netD_B_predict_fake)
    else:
        G_B_target = K.ones_like(netD_A_predict_fake)
        G_A_target = K.ones_like(netD_B_predict_fake)

    loss_G_B = criterion_GAN(netD_A_predict_fake, G_B_target)
    loss_G_A = criterion_GAN(netD_B_predict_fake, G_A_target)

    loss_GAN = loss_G_A + loss_G_B

    # Cycle loss
    loss_cyc = compute_similarity_loss(rec_A, real_A, rec_B, real_B)

    # Identity loss
    loss_id = compute_similarity_loss(fake_B, real_A, fake_A, real_B)

    loss_G = loss_GAN + cycle_loss_weight * loss_cyc + id_loss_weight * loss_id

    return loss_G


def discriminator_loss(netD_predict, use_wgan=False):
    netD_predict_real, netD_predict_fake = netD_predict

    if use_wgan:
        real_target = -K.ones_like(netD_predict_real)
        fake_target = K.ones_like(netD_predict_fake)
    else:
        real_target = K.ones_like(netD_predict_real)
        fake_target = K.zeros_like(netD_predict_fake)

    netD_loss_real = criterion_GAN(netD_predict_real, real_target)
    netD_loss_fake = criterion_GAN(netD_predict_fake, fake_target)

    loss_netD = (1 / 2) * (netD_loss_real + netD_loss_fake)
    return loss_netD
