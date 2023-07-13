import math 

class microLIF:
    def __init__(self, input_features, output_features, dt, tau_mem=10e-3, tau_syn=5e-3,
                 kernel_initializer="random_normal",
                 monitor="mem", initial_state=0, use_bias=False, reset=False, **args):
        self.input_features = input_features
        self.output_features = output_features
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.tau_mem_w = tau_mem
        self.tau_syn_w = tau_syn
        self.syn = None
        self.mem = None
        self.reset = reset
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.initial_state = initial_state
        self.monitor = monitor
        self.eta = args.get("eta", 0.5)
        self.hebbian = args.get("hebbian", False)
        self.threshold = 1.0
        scale = 4.0
        self.w = [[0.0]*self.output_features]*self.input_features

        if self.kernel_initializer == "he_normal":
            self.w = [[scale]*self.output_features]*self.input_features
        if self.kernel_initializer == "zeros":
            self.w = [[0.0]*self.output_features]*self.input_features
        if self.kernel_initializer == "ones":
            self.w = [[1.0]*self.output_features]*self.input_features
        if self.use_bias:
            self.bias = [0.0]*self.output_features

    def set_tau(self, tau_mem, tau_syn):
        self.tau_mem_w = tau_mem
        self.tau_syn_w = tau_syn

    # For now, the activation function is a simple step function
    def activation(self, mthr):
        return 1.0 if mthr > 0 else 0.0

    def get_spike_and_reset(self, mem):
        mthr = mem - self.threshold
        out = self.activation(mthr)
        rst = out
        return out, rst
    
    def reset_states(self):
        self.syn = None
        self.mem = None

    def dot_product(a, b):
        return sum([a[i]*b[i] for i in range(len(a))])

    def simulate(self, inputs):
        # multiply feed forward
        input = [self.dot_product(inputs[i], self.w[i]) for i in range(len(inputs))]
        if self.use_bias:
            input = [input[i] + self.bias[i] for i in range(len(input))]

        ops_entry = self.dot_product([float(i != 0) for i in inputs], [float(w != 0) for w in self.w])

        # stats
        nb_steps = len(input)

        # variables
        if self.syn is None or self.reset_states:
            syn_tf = [0 for _ in range(len(input))]
            self.syn = syn_tf[0]
        if self.mem is None or self.reset_states:
            mem_tf = [i * self.initial_state for i in input]
            self.mem = mem_tf[0]

        self.dcy_mem = math.exp(-self.dt / (self.tau_mem_w + 1e-16))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn =  math.exp(-self.dt / (self.tau_syn_w + 1e-16))
        self.scl_syn = 1.0 - self.dcy_syn

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_store = []
        syn_store = []
        out_store = []

        # first step
        new_mem = input[0] * self.initial_state
        new_syn = input[0] * 0
        new_out = input[0] * 0

        mem_store.append(new_mem)
        syn_store.append(new_syn)
        out_store.append(new_out)

        # Here we loop over time
        for t in range(nb_steps - 1):
            # synaptic & membrane dynamics
            new_syn = self.dcy_syn * self.syn + input[t]
            new_mem = self.dcy_mem * self.mem + self.scl_mem * self.syn  # multiplicative reset

            if self.activation:
                new_out, rst = self.get_spike_and_reset(self.mem)
                new_mem = new_mem * (1.0 - rst)
            else:
                new_out = new_mem

            self.syn = new_syn
            self.mem = new_mem

            mem_store.append(new_mem)
            syn_store.append(new_syn)
            out_store.append(new_out)

        return mem_store, syn_store, out_store, ops_entry
    
    def forward(self, inputs):

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