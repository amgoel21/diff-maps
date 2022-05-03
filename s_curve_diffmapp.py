import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numbers
import array
from collections.abc import Iterable
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from sklearn.datasets import  make_s_curve
import numpy as np
import matplotlib.pyplot as plt

def convertdata(data):
    x_data = torch.tensor(data)
    return x_data

def diffusion_maps_loss(z, args):
    epsilon = 0.35
    with torch.no_grad():
        znew = z.unsqueeze(0)
        k = torch.exp(-torch.cdist(znew, znew, p=2) ** 2 / epsilon).squeeze(0)   # kernel
        
        d = torch.sum(k, dim=-1)   # diagonal
        if args[0] > 0.0:
            diag_alpha = torch.pow(1 / d, args[0]).view(1, -1)   # raise to alpha
            k = diag_alpha.T * k * diag_alpha   # conjugation
        
        k.fill_diagonal_(0)
        d_alpha = 1 / torch.sum(k, dim=-1)
        sq_dalpha = torch.pow(d_alpha, 1/2).view(1, -1)
        m = -sq_dalpha.T * k * sq_dalpha    
        m.fill_diagonal_(1)

        #w, v = torch.linalg.eigh(m)
        w,v = torch.lobpcg(-m,10) # smallest
        print(w)
        embedding = v[:,1:args[1] + 1] * sq_dalpha.T
    return embedding

if __name__ == '__main__':
    X,Y = make_s_curve(2000)
    data=convertdata(X)
    embedding = diffusion_maps_loss(data,[1,20])
    plt.scatter(embedding[:,0],embedding[:,1], c=np.concatenate([Y],axis=0))
plt.show()