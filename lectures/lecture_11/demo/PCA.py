
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# A simple example of PCA
from pylab import *
from numpy import *
from scipy import linalg as la
from matplotlib import *
#import pyplot as plt
import os 

iris  = loadtxt('iris_proc.data', delimiter = ',')

def pca(data,nRedDim=0,normalise=1):
    
    # Centre data
    m = mean(data,axis=0)
    data -= m

    # Covariance matrix
    C = cov(transpose(data))

    # Compute eigenvalues and sort into descending order
    evals,evecs = linalg.eig(C) 
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]

    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
    
    if normalise:
        for i in range(shape(evecs)[1]):
           evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
          
    # Produce the new data matrix
    z = dot(transpose(evecs),transpose(data))
    x = transpose(z)
    
    # Compute the original data again
    y=transpose(dot(evecs,z))
    y += m
    data += m
       
    return x,y,evals,evecs,m
# d - the first 4 components of 
d =  iris[:,:4]   

print('d=', d)
x,y,evals,evecs,m = pca(d,2)


plt.plot(iris[:,0],iris[:,1],'o')
plt.title('Original dataset')
plt.axis('off')
plt.figure(2)

plt.plot(x[:,0], x[:,1],'ok',markersize=7, markerfacecolor='red')
plt.title('New components z')
plt.axis('off')
plt.figure(3)

plt.plot(y[:,0],y[:,1],'ok', markersize=7, markerfacecolor='blue')
plt.title('Reconstructed data after PCA')
plt.axis('off')
plt.show()

plt.show()



