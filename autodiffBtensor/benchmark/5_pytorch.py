import torch
torch.set_printoptions(linewidth=200)

def computation(p):
    inp2 = torch.cos(p)
    final = torch.sum(inp2, 1)
    return final


def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())

test = torch.tensor([[1,2,3],
                     [4,5,6],
                     [7,8,9]], dtype=torch.float64, requires_grad=True)
#
#result = computation(test)
#g = jacobian(result, test, create_graph=True)
#g = g.reshape(3,9)
#h = jacobian(g, test, create_graph=True)
#h = h.reshape(3,9,9)
#c = jacobian(h, test, create_graph=True)
#c = c.reshape(3,9,9,9)

def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data


r = get_jacobian(computation, test.flatten(), 3)
print(r)
