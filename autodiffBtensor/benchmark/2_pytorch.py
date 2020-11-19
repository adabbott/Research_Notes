import torch
torch.set_printoptions(linewidth=200)


def computation(p):
    inp2 = torch.cos(p)
    final = torch.sum(inp2, 1)
    return final

test = torch.tensor([[1,2,3],
                     [4,5,6],
                     [7,8,9]], dtype=torch.float64, requires_grad=True)
result = computation(test)
print(result)

def derivatives(inp, order=1):
    """Computes the n'th order derivative"""
    out = computation(inp)
    dim1 = out.shape[0]
    dim2 = inp.flatten().shape[0]
    shape = [dim1, dim2]
    count = 0
    while count < order:
        derivatives = []
        for val in out.flatten():
            d = torch.autograd.grad(val, inp, create_graph=True)[0].reshape(dim2)
            derivatives.append(d)
        out = torch.stack(derivatives).reshape(tuple(shape))
        shape.append(dim2)
        count += 1
    return out

#print(derivatives(test, order=1))
#print(derivatives(test, order=1))
a = derivatives(test, order=3)
#a = derivatives(test, order=3)
print(a)
print(a.shape)


