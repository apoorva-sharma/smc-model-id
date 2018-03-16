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
        selected_pred = tf.reduce_sum(mask*pred, axis=0)
        loss = tf.reduce_mean( tf.nn.l2_loss(selected_pred - targets) )

    return loss



class MultiHeadDynModel:
    def __init__(self, x_dim, u_dim, layer_sizes, n_heads, name="model", trial=0):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.layer_sizes = layer_sizes
        self.n_heads = n_heads


        self.lr = 2e-4
        self.beta1 = 0.9
        self.batch_size = 50

        self.save_path = name + "/trial" + str(trial)

    def build_model(self):
        self.x_ = x_ = tf.placeholder(tf.float32, shape=(None, self.x_dim))
        self.u_ = u_ = tf.placeholder(tf.float32, shape=(None, self.u_dim))
        self.xp_ = xp_ = tf.placeholder(tf.float32, shape=(None, self.x_dim))
        self.i_ = index_ = tf.placeholder(tf.int32, shape=(None,))

        self.x_pred = x_pred = multi_head_predictor(x_,u_,self.layer_sizes,self.n_heads)

        self.loss = multi_head_loss(x_pred, xp_, index_, self.n_heads)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        optim = tf.train.AdamOptimizer(self.lr, self.beta1, name="optim")
        self.train_op = optim.minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def init(self):
        tf.global_variables_initializer().run()

    def train(self, x_input, u_input, xp_input, ind_input, verbose=False):
        N = x_input.shape[0]
        B = self.batch_size
        sess = tf.get_default_session()
        self.writer = tf.summary.FileWriter(self.save_path + '/train', sess.graph)
        for epoch in range(500):
            random_idx = np.random.permutation(N)
            num_batches = N // B
            for i in range(num_batches):
                batch_idx = random_idx[range(i*B,(i+1)*B)]
                x_batch = x_input[batch_idx,:]
                u_batch = u_input[batch_idx,:]
                xp_batch = xp_input[batch_idx,:]
                ind_batch = ind_input[batch_idx]

                inputs = {self.x_: x_batch,
                          self.u_: u_batch,
                          self.xp_: xp_batch,
                          self.i_: ind_batch}

                _, loss = sess.run((self.train_op, self.loss), feed_dict=inputs)

            if epoch % 5 == 0 and verbose:
                print('Epoch',epoch,'\tBatch',i,'\tLoss:',loss)

        print("Final Loss: ", loss)

    def predict(self, x_test, u_test):
        sess = tf.get_default_session()
        return sess.run(self.x_pred, {self.x_: x_test, self.u_: u_test})

    def save(self):
        sess = tf.get_default_session()
        self.saver.save(sess, self.save_path, global_step=self.global_step)

    def load(self):
        sess = tf.get_default_session()
        path = tf.train.latest_checkpoint(self.save_path)
        self.saver.restore(sess, path)
