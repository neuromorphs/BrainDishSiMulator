#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 30.03.23
Description: Few initializers for the spiking neural network
"""
import tensorflow as tf
import numpy as np

def get_lif_kernel(tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, max_duration=1.0):
    """ Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.
    Args:
        tau_mem: The membrane time constant
        tau_syn: The synaptic time constant
        dt: The timestep size
    Returns:
        Array of length 10x of the longest time constant containing the filter kernel
    """

    tau_max = np.max((tau_mem, tau_syn))
    kernel_length = 10
    scale = 1e3*max_duration * dt * kernel_length

    # clip tau_max if it is too large
    dcy1 = np.exp(-dt / (tau_mem + 1e-16))  # handle case tau_mem = 0
    dcy2 = np.exp(-dt / (tau_syn + 1e-16))  # handle case tau_syn = 0

    ts = np.arange(0, int(tau_max*scale/dt))*dt
    n = len(ts)
    kernel = np.empty(n)
    I = 1.0  # Initialize current variable for single spike input
    U = 0.0

    for i, t in enumerate(ts):
        kernel[i] = U
        U = dcy1*U + (1.0-dcy1)*I
        I *= dcy2
    return kernel


def _get_epsilon(calc_mode, tau_mem, tau_syn, timestep=1e-3):
    if calc_mode == 'analytical':
        return _epsilon_analytical(tau_mem, tau_syn)

    elif calc_mode == 'numerical':
        return _epsilon_numerical(tau_mem, tau_syn, timestep)

    else:
        raise ValueError('invalid calc mode for epsilon')

def _epsilon_analytical(tau_mem, tau_syn):
    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn ** 2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat

def _epsilon_numerical(tau_mem, tau_syn, timestep):
    # case IAF
    if tau_syn == 0 and tau_mem == np.inf:
        return 1, 1

    kernel = get_lif_kernel(tau_mem, tau_syn, timestep)
    epsilon_bar = kernel.sum() * timestep
    epsilon_hat = (kernel**2).sum() * timestep

    return epsilon_bar, epsilon_hat


class FluctuationDrivenInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mu_u,
                 sigma_u,
                 nu,
                 tau_mem,
                 tau_syn,
                 timestep,
                 epsilon_calc_mode='numerical',
                 seed=None,
                 **kwargs):
        """
        Fluctuation-driven initialization for the weights of a spiking neural network.
        :param mu_u: target mean membrane potential
        :param sigma_u: target standard deviation of membrane potential
        :param nu: mean input firing rate
        :param tau_mem: time constant of the membrane
        :param tau_syn: time constant of the synapses
        :param timestep: resolution of timestep [s]
        :param epsilon_calc_mode: calculation mode for the epsilon parameters, either 'analytical' or 'numerical'. Default is 'numerical'.
        :param seed: seed for the random normal number generator
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.mu_u = mu_u
        self.xi = 1 / sigma_u
        self.sigma_u = sigma_u
        self.nu = nu
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.timestep = timestep
        self.epsilon_calc_mode = epsilon_calc_mode
        self.seed = seed

        self.ebar, self.ehat = _get_epsilon(self.epsilon_calc_mode,
                                  self.tau_mem,
                                  self.tau_syn,
                                  self.timestep)

    def __call__(self, shape, dtype=None):
        # Implement the initialization logic here
        # The `shape` argument specifies the shape of the weight tensor being initialized
        # The `dtype` argument specifies the data type of the weights
        # This method should return a tensor of the specified shape and data type
        theta = 1.0  # default threshold

        n = shape[0]

        mu_w = self.mu_u / (n * self.nu * self.ebar)
        sigma_w = tf.math.sqrt(1 / (n * self.nu * self.ehat) * ((theta - self.mu_u) / self.xi) ** 2 - mu_w ** 2)

        mu_w = tf.cast(mu_w, dtype)
        sigma_w = tf.cast(sigma_w, dtype)

        weights = tf.random.normal(shape, mu_w, sigma_w, seed=self.seed, dtype=dtype)
        return weights

    def get_config(self):
        # Return a dictionary of the initializer's hyperparameters and variables
        # This is necessary for serialization and deserialization
        config = {
            'mu_u': self.mu_u,
            'sigma_u': self.sigma_u,
            'nu': self.nu,
            'tau_mem': self.tau_mem,
            'tau_syn': self.tau_syn,
            'timestep': self.timestep,
            'epsilon_calc_mode': self.epsilon_calc_mode,
            'seed': self.seed
        }
        return config
