# Numpy implementation of Adap optimizer from scratch.
# Borrowed from https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
# and https://gist.github.com/hrayrhar/3b809c5ae778485a9ea9d253c4bfc90a

import numpy as np


class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., ):

        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def update(self, params, grads):
        """
        Perform a gradient update step.
        :param params: list of numpy arrays
        :param grads: list of numpy arrays; same shapes as params
        :return: list of updated parameter arrays
        """

        """ #TODO: implement clipping
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        """

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros_like(p) for p in params]
            self.vs = [np.zeros_like(p) for p in params]

        updated_params = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            updated_params[i] = p_t

        self.iterations += 1

        return updated_params
