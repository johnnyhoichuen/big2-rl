import tensorflow as tf
import numpy as np
from baselines.a2c.utils import fc
import joblib

import torch
from big2_rl.env.env import get_obs

sess = tf.Session()


class PPONetwork(object):

    def __init__(self, sess, obs_dim, act_dim, name):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name

        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, obs_dim], name="input")
            available_moves = tf.placeholder(tf.float32, [None, act_dim], name="availableActions")
            # available_moves takes form [0, 0, -inf, 0, -inf...], 0 if action is available, -inf if not.
            activation = tf.nn.relu
            h1 = activation(fc(X, 'fc1', nh=512, init_scale=np.sqrt(2)))
            h2 = activation(fc(h1, 'fc2', nh=256, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', act_dim, init_scale=0.01)
            # value function - share layer h1
            h3 = activation(fc(h1, 'fc3', nh=256, init_scale=np.sqrt(2)))
            vf = fc(h3, 'vf', 1)[:, 0]
        availPi = tf.add(pi, available_moves)

        def sample():
            u = tf.random_uniform(tf.shape(availPi))
            return tf.argmax(availPi - tf.log(-tf.log(u)), axis=-1)

        a0 = sample()
        el0in = tf.exp(availPi - tf.reduce_max(availPi, axis=-1, keep_dims=True))
        z0in = tf.reduce_sum(el0in, axis=-1, keep_dims=True)
        p0in = el0in / z0in
        onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))

        def step(obs, availAcs):
            a, v, neglogp = sess.run([a0, vf, neglogpac], {X: obs, available_moves: availAcs})
            return a, v, neglogp

        def value(obs, availAcs):
            return sess.run(vf, {X: obs, available_moves: availAcs})

        self.availPi = availPi
        self.neglogpac = neglogpac
        self.X = X
        self.available_moves = available_moves
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def getParams():
            return sess.run(self.params)

        self.getParams = getParams

        def loadParams(paramsToLoad):
            restores = []
            for p, loadedP in zip(self.params, paramsToLoad):
                restores.append(p.assign(loadedP))
            sess.run(restores)

        self.loadParams = loadParams

        def saveParams(path):
            modelParams = sess.run(self.params)
            joblib.dump(modelParams, path)

        self.saveParams = saveParams


class PPOAgent:
    def __init__(self, model_path):
        """
        Loads model's pretrained weights from a given model path.
        """
        self.model = PPONetwork(sess, 412, 1695, "trainNet")
        params = joblib.load("modelParameters136500")
        self.model.loadParams(params)

    def act(self, infoset):
        """
        Given an infoset available to this agent, takes the z_batch and x_batch features (historical actions
        and current state + action) and computes forward pass of model to get the suggested legal action.
        However, if only one action is legal (pass), then take that action.
        """
        if len(infoset.legal_actions) == 1:  # default case
            return infoset.legal_actions[0]
        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values'].detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action
