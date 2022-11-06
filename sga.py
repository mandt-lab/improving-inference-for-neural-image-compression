# Standalone implementation of SGA (Stochastic Gumbel Annealing).

import tensorflow as tf
import tensorflow_probability as tfp



@tf.function
def sga_round_no_offset(mu: tf.Tensor, tau: float, epsilon: float = 1e-5):
  """Draw one sample from the stochastic rounding distribution defined by SGA.

  This can be seen as stochastically rounding the variational parameter mu,
  which we usually initialize to the prediction of an amortized inference
  network.
  This function rounds to the usual integer ground, hence "no_offset".
  Args:
    mu:  "location" variational parameter of the stochastic rounding
      distribution in SGA.
    tau: temperature of the rounding distribution, as well as in the
      Gumbel-softmax trick for sampling.
    epsilon: small constant for numerical stability

  Returns:
    stoch_round_sample: SGA sample
  """
  mu_floor = tf.math.floor(mu)
  mu_ceil = tf.math.ceil(mu)
  mu_bds = tf.stack([mu_floor, mu_ceil], axis=-1)
  round_dir_dist_logits = tf.stack(
    [
      -tf.math.atanh(
        tf.clip_by_value(mu - mu_floor, -1 + epsilon, 1 - epsilon)) / tau,
      -tf.math.atanh(
        tf.clip_by_value(mu_ceil - mu, -1 + epsilon, 1 - epsilon)) / tau
    ],
    axis=-1)  # last dim are logits for DOWN or UP
  # Create a Concrete distribution of the rounding direction r.v.s.
  round_dir_dist = tfp.distributions.RelaxedOneHotCategorical(
    tau, logits=round_dir_dist_logits
  )  # We can use a different temperature here, but it hasn't been explored.
  round_dir_sample = round_dir_dist.sample()
  stoch_round_sample = tf.reduce_sum(
    mu_bds * round_dir_sample, axis=-1)  # inner product in last dim
  return stoch_round_sample


@tf.function
def sga_round(mu: tf.Tensor, tau: float, offset=None, epsilon: float = 1e-5):
  """
  Same as sga_round_no_offset but allow rounding to a shifted integer grid.
  """
  if offset is None:
    return sga_round_no_offset(mu, tau, epsilon)
  else:
    return sga_round_no_offset(mu - offset, tau, epsilon) + offset


def get_sga_schedule(r, ub, lb=1e-8, scheme='exp', t0=200.0):
  """Get the annealing schedule for the temperature (tau) param in SGA.

  Args:
    r: decay strength
    ub: maximum/init temperature
    lb: small const like 1e-8 to prevent numerical issue when temperature too
      close to 0
    scheme:
    t0: the number of "warmup" iterations, during which the temperature is fixed
      at the value ub.

  Returns:
    callable t -> tau(t)
  """
  backend = tf

  def schedule(t):
    # :param t: step/iteration number
    t = tf.cast(t, tf.float32)  # step variable is usually tf.int64
    if scheme == 'exp':
      tau = ub * backend.exp(-r * (t - t0))
    elif scheme == 'linear':
      # Cool temperature linearly from ub after the initial t0 iterations
      tau = -r * (t - t0) + ub
    else:
      raise NotImplementedError

    if backend is None:
      return min(max(tau, lb), ub)
    else:
      return backend.minimum(backend.maximum(tau, lb), ub)

  return schedule


default_sga_schedule = get_sga_schedule(
  r=1e-3, ub=0.5, scheme='exp', t0=700.0)  # default from paper img experiments

