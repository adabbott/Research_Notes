import numpy as np

def get_architectures(layers, inp_dim):
    """
    Takes a list of hidden layer tuples (n,n,n...) and input dimension size and
    sorts it by number of weights in the neural network
    """
    out_dim = 1
    sizes = []
    for struct in layers:
        size = 0
        idx = 0 
        size += inp_dim * struct[idx]
        idx += 1
        n = len(struct)
        while idx < n:
            size += struct[idx - 1] * struct[idx]
            idx += 1
        size += out_dim * struct[-1] 
        sizes.append(size)
    sorted_indices = np.argsort(sizes).tolist()
    layers = np.asarray(layers)
    layers = layers[sorted_indices].tolist()
    return layers

layers =[(20,), (20,20), (20,20,20), (20,20,20,20), (20,20,20,20,20),
         (40,), (40,40), (40,40,40), (40,40,40,40), (40,40,40,40,40),
         (60,), (60,60), (60,60,60), (60,60,60,60), (60,60,60,60,60),
         (80,), (80,80), (80,80,80), (80,80,80,80), (80,80,80,80,80)]

current_layers = get_architectures(layers, 3)
print(current_layers)


