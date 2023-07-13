import warnings
import torch
import torch.nn as nn
from .activations import SuperSpike

class LIF(nn.Module):
    def __init__(self, input_features, output_features, dt, tau_mem=10e-3, tau_syn=5e-3,
                 kernel_initializer="random_normal",
                 monitor="mem", initial_state=0, use_bias=False, reset=False, **args):
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

        self.eta = args.get("eta", 0.5)
        self.hebbian = args.get("hebbian", False)
        
        self.A_plus = torch.tensor([.1])
        self.A_minus = torch.tensor([- self.A_plus])
        self.tau_plus = torch.tensor([10.])
        self.tau_minus = torch.tensor([10.])
        

        if monitor != "out":
            warnings.warn("Monitoring {} instead of output spikes".format(monitor))

        self.threshold = 1.0

        scale = 4.0
        if self.hebbian :
            self.r_grad = False
        else :
            self.r_grad = True 
        
        self.w = nn.Parameter(torch.randn((self.input_features, self.output_features), requires_grad=self.r_grad)*scale)

        if self.kernel_initializer == "he_normal":
            nn.init.kaiming_uniform_(self.w, a=0, mode='fan_in', nonlinearity='relu')
        if self.kernel_initializer == "zeros":
            self.w = nn.Parameter(torch.zeros((self.input_features, self.output_features), requires_grad=self.r_grad))
        if self.kernel_initializer == "ones":
            self.w = nn.Parameter(torch.ones((self.input_features, self.output_features), requires_grad=self.r_grad))

        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(self.output_features, requires_grad=self.r_grad))

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

        self.dcy_mem = torch.exp(torch.tensor(-self.dt / (self.tau_mem_w + 1e-16)))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn =  torch.exp(torch.tensor(-self.dt / (self.tau_syn_w + 1e-16)))
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
            
            if self.hebbian:  # Assuming you use this to decide whether to use STDP or not
                '''for i in range(self.input_features):
                    #print(i)
                    for j in range(self.output_features):
                        #print(j)
                        # Compute STDP update for each pair of pre and post-synaptic neurons
                        delta_w = self.compute_stdp_update(inputs[:, t, i], new_out[:, j])
                        #print(delta_w)
                        #print(self.w.shape)
                        self.w[i, j] += delta_w.squeeze()'''
                        
                for i in range(self.input_features):
                    for j in range(self.output_features):
                        # Compute STDP update for each pair of pre and post-synaptic neurons
                        delta_t = inputs[:, t, i] - new_out[:, j]
                        delta_w = self.compute_complex_stdp_update(inputs[:, t, i].unsqueeze(0), new_out[:, j].unsqueeze(0), delta_t)
                        #print(delta_w.shape)
                        #print(self.w.shape)
                        self.w += delta_w #.squeeze()

            mem_store.append(new_mem)
            syn_store.append(new_syn)
            out_store.append(new_out)

        mem_store = torch.stack(mem_store, dim=1)
        syn_store = torch.stack(syn_store, dim=1)
        out_store = torch.stack(out_store, dim=1)

        return mem_store, syn_store, out_store, ops_entry

    def get_weights(self):
        return self.w.clone()
    
    def get_spikes(self):
        return self.spikes.clone()

    def reset_states(self):
        self.syn = None
        self.mem = None

    def forward(self, inputs):
        
        if self.hebbian :
            with torch.no_grad() :
                U, I, O, ops = self.simulate(inputs)
                self.spikes = O[:,-1]
        else :
            U, I, O, ops = self.simulate(inputs)
        """
        if self.hebbian:
            pre_synaptic_activity = inputs
            post_synaptic_activity = O

            if post_synaptic_activity.sum()==0: # no spikes
                self.w += 0.1
            else:
                self.w += self.eta * torch.mm(pre_synaptic_activity.t(), post_synaptic_activity)
        """
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
    
    def compute_stdp_update(self, pre_syn_activity, post_syn_activity):
        # Compute the weight update from the STDP rule
        # This will depend on the precise form of your STDP rule
        # For simplicity, let's say that we increase the weight if both pre and post activity exist, 
        # and decrease it otherwise. This is not a biologically accurate STDP rule, but is just for example.
        return self.eta * (pre_syn_activity - post_syn_activity)

    def compute_complex_stdp_update(self, pre_spike, post_spike, delta_t):
        # Initialize weight update
        delta_w = torch.zeros_like(self.w)

        for t in range(delta_t.size(0)):
            #print(pre_spike[t].t().shape)
            #print(post_spike[t].shape)

            if delta_t[t] > 0:
                # If post_spike occurred after pre_spike, potentiate synapse
                #delta_w += self.A_plus * torch.exp(-delta_t[t] / self.tau_plus) * torch.mm(pre_spike[t].t(), post_spike[t])
                delta_w += self.A_plus * torch.exp(delta_t[t] / self.tau_plus) * torch.outer(pre_spike[t], post_spike[t])

            else:
                # If post_spike occurred before pre_spike, depress synapse
                delta_w -= self.A_minus * torch.exp(delta_t[t] / self.tau_minus) * torch.outer(pre_spike[t], post_spike[t])

        return delta_w
