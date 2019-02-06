from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model

from .networks_utils import (residual_dense_block,
                             dense_layer)


def resnet_generator_FC_bigger(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                               use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = dense_layer(embedding, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    embedding = dense_layer(embedding, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator_FC_smaller(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator_FC_smallest(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator(network_type='FC_smaller', **args):
    assert network_type in {'FC_smaller', 'FC_smallest', 'FC_bigger'}, "NOT IMPLEMENTED FOR THIS 'network_type'!!!"

    generators = {
        "FC_smaller": resnet_generator_FC_smaller,
        "FC_smallest": resnet_generator_FC_smallest,
        "FC_bigger": resnet_generator_FC_bigger,
    }

    return generators[network_type](**args)
