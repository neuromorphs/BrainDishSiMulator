import torch

class SuperSpike(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta.
    """
    beta = 20.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.beta*torch.abs(input)+1.0)**2
        return grad