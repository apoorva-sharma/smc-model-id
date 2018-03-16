import time
import tensorflow as tf
import numpy as np
# x \in R^n
# u \in R^m
# batch_size = N

# x, u, are tensors with dim [N,n] and [N,m] respectively
def multi_head_predictor(x, u, layer_sizes=[64,64,8], n_heads=50, name="multihead"):
    with tf.variable_scope(name):
        x_dim = x.get_shape().as_list()[-1]
        input_layer = tf.concat([x, u], axis=-1)
        current_input = input_layer
        for l,units in enumerate(layer_sizes):
            current_input = tf.layers.dense(
                                inputs=current_input,
                                units=units,
                                activation=tf.nn.relu)

        pred_list = [tf.layers.dense(
                        inputs=current_input,
                        units=x_dim,
                        activation=None,
                        name="head"+str(i+1)) for i in range(n_heads)]

        pred_tensor = tf.stack(pred_list, axis=0)

    return pred_tensor

# pred: (M, N, n)
# targets: (N, n)
# ind: (N)
def multi_head_loss(pred, targets, ind, n_heads, name="loss"):
    with tf.variable_scope(name):
        mask = tf.expand_dims( tf.one_hot(ind, n_heads, axis=0), axis=-1)
        print(mask.get_shape().as_list())
        selected_pred = tf.reduce_sum(mask*pred, axis=0)
        loss = tf.reduce_mean( tf.nn.l2_loss(selected_pred - targets) )

    return loss
