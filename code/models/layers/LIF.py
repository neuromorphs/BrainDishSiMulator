import warnings
import torch
import torch.nn as nn
from .activations import SuperSpike

class LIF(nn.Module):
    def __init__(self, input_features, output_features, dt, tau_mem=10e-3, tau_syn=5e-3,
                 kernel_initializer="he_normal",
                 monitor="out", initial_state=0, use_bias=False, reset=False, **args):
        """
        Simple feed-fordward spiking layer for spiking neural networks
        """
        super(LIF, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.dt = dt

        self.tau_mem = tau_mem
        if tau_syn is None:
            tau_syn = 0.0
        self.tau_syn = tau_syn
        self.tau_mem_w = tau_mem
        self.tau_syn_w = tau_syn
        self.syn = None
        self.mem = None
        self.reset = reset

        self.activation = SuperSpike
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.initial_state = initial_state
        self.monitor = monitor

        if monitor != "out":
            warnings.warn("Monitoring {} instead of output spikes".format(monitor))

        self.threshold = 1.0

        self.w = nn.Parameter(torch.randn((self.input_features, self.output_features), requires_grad=True))

        if self.kernel_initializer == "he_normal":
            nn.init.kaiming_uniform_(self.w, a=0, mode='fan_in', nonlinearity='relu')
        if self.kernel_initializer == "zeros":
            self.w = nn.Parameter(torch.zeros((self.input_features, self.output_features), requires_grad=True))


        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_features, requires_grad=True))

    def set_tau(self, tau_mem, tau_syn):
        self.tau_mem_w = tau_mem
        self.tau_syn_w = tau_syn

    def get_spike_and_reset(self, mem):
        mthr = mem - self.threshold
        out = self.activation.apply(mthr)
        rst = out
        return out, rst

    def simulate(self, inputs):
        # multiply feed forward
        input = torch.matmul(inputs, self.w)
        if self.use_bias:
            input = input + self.bias

        ops_entry = torch.matmul((inputs != 0).float(), (self.w != 0).float())

        # stats
        nb_steps = input.shape[1]

        # variables
        if self.syn is None or self.reset_states:
            syn_tf = input * 0
            self.syn = syn_tf[:, 0, :]
        if self.mem is None or self.reset_states:
            mem_tf = input * self.initial_state
            self.mem = mem_tf[:, 0, :]

        self.dcy_mem = torch.math.exp(-self.dt / (self.tau_mem_w + 1e-16))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn =  torch.math.exp(-self.dt / (self.tau_syn_w + 1e-16))
        self.scl_syn = 1.0 - self.dcy_syn

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_store = []
        syn_store = []
        out_store = []

        # first step
        new_mem = input[:, 0] * self.initial_state
        new_syn = input[:, 0] * 0
        new_out = input[:, 0] * 0

        mem_store.append(new_mem)
        syn_store.append(new_syn)
        out_store.append(new_out)

        # Here we loop over time
        for t in range(nb_steps - 1):
            # synaptic & membrane dynamics
            new_syn = self.dcy_syn * self.syn + input[:, t]
            new_mem = (self.dcy_mem * self.mem + self.scl_mem * self.syn)  # multiplicative reset

            if self.activation:
                new_out, rst = self.get_spike_and_reset(self.mem)
                new_mem = new_mem * (1.0 - rst)
            else:
                new_out = new_mem

            self.syn = new_syn.clone()
            self.mem = new_mem.clone()

            mem_store.append(new_mem)
            syn_store.append(new_syn)
            out_store.append(new_out)

        mem_store = torch.stack(mem_store, dim=1)
        syn_store = torch.stack(syn_store, dim=1)
        out_store = torch.stack(out_store, dim=1)

        return mem_store, syn_store, out_store, ops_entry

    def get_weights(self):
        return self.w.clone()

    def reset_states(self):
        self.syn = None
        self.mem = None

    def forward(self, inputs):
<<<<<<< Updated upstream
=======
        
        if self.hebbian :
            with torch.no_grad() :
                U, I, O, ops = self.simulate(inputs)
                self.spikes = O[:,-1]
        else :
            U, I, O, ops = self.simulate(inputs)
            self.spikes = O[:,-1]
        """
        if self.hebbian:
            pre_synaptic_activity = inputs
            post_synaptic_activity = O
>>>>>>> Stashed changes

        U, I, O, ops = self.simulate(inputs)

        if self.monitor == "mem":
            return_value = U
        elif self.monitor == "syn":
            return_value = I
        elif self.monitor == "out":
            return_value = O
        elif self.monitor == "ops":
            return_value = ops
        else:  # default
            raise ValueError("monitor must be one of 'mem', 'syn', 'out', 'ops'")

        return return_value