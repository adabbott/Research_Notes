#import numpy as np
#import GPy
#import json
#import re
#from itertools import combinations

# You must use exact same morse alpha and sklearn scaling procedure. In this case, the means, variances/scales from sklearn standard scaler for X only are needed
#Best performing hyperparameters are:
#[('fi_transform', {'degree_reduction': False, 'fi': True}), ('morse_transform', {'morse': True, 'morse_alpha': 1.1116282254869856}), ('scale_X', 'std'), ('scale_y', None)]
#Best model performance (cm-1):
#Means  [0.22469701 0.78085142 0.30732701]
#Scales  [0.02572785 0.05263419 0.04131716]
#Test Dataset 17.26
#Full Dataset 15.35

# point 25 from pes_data, point 9 in MLChem arrays cuz of energy ordering
# H H O ordering
import numpy as np
cart_vector = np.array(
 [0.0000000000, 0.0000000000, 0.9500000000,
  0.0000000000, 0.9848077530, -0.1736481777,
  0.0000000000, 0.0000000000, 0.0000000000 ])


def PES(cart_vector):
    """
    Given cartesian coordinates in proper atom order (MLChem standard order), computes the energy
    """
    import numpy as np
    import GPy
    import json
    import re
    from itertools import combinations


    # needs to be in standard order if not already
    natoms = 3  # hard coded
    cart = cart_vector.reshape(natoms, 3)
    newcart = np.zeros((cart.shape[0],cart.shape[0] )) 
    for i,j in combinations(range(len(cart)),2):
        R = np.linalg.norm(cart[i]-cart[j])         
        newcart[j,i] = R                             
    idm = newcart[np.tril_indices(len(newcart),-1)]

    # transform geometry according to MLChem.guassian_process.GaussianProcess.preprocess
    # Morse variable
    morse = np.exp(-idm/1.111628225)  # hard coded
    # Fundamental invariants
    polys = ['x0', 'x1+x2', 'x1**2+x2**2']               # hard coded
    new = np.zeros((len(polys)))                         # hard coded                                                    
    for i, p in enumerate(polys):                        # hard coded              
        eval_string = re.sub(r"(x)(\d+)", r"morse[\2]", p) # hard coded                                           
        new[i] = eval(eval_string)                       # hard coded                                            
    
    # standard scaling
    # means/"scales" need to be taken directly from a run with exact hyperparameters 
    # since they change based on every hp combination
    means =  np.array([0.22469701, 0.78085142, 0.30732701])
    scales = np.array([0.02572785, 0.05263419, 0.04131716])
    new -= means 
    new /= scales 
    

    # load GPy
    model = GPy.core.model.Model('mymodel')
    with open('model.json', 'r') as f:
        model_dict = json.load(f)
    final = model.from_dict(model_dict) 
    # GPy needs a column vector
    new = np.expand_dims(new, axis=0)
    e, cov = final.predict(new)
    energy = e[0,0]
    return energy
    
    
energy = PES(cart_vector)
print(energy)
