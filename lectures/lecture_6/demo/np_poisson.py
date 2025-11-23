import numpy as np
from timeit import default_timer as timer

def naive_update(u,dx,dy):
    [nx,ny] = u.shape
    dx2 = dx**2
    dy2 = dy**2
    u_old = np.copy(u)
    for i in range(1,nx-1):
        for j in range(1, ny-1):
            u[i,j] = (  (u_old[i+1,j  ] + u_old[i-1,j  ])*dy2  \
                      + (u_old[i  ,j+1] + u_old[i  ,j-1])*dx2) \
                   / (2*(dx2 + dy2))

def numpy_update(u,dx,dy):
    dx2 = dx**2
    dy2 = dy**2
    u_old = np.copy(u)
    u[1:-1,1:-1] = (  (u_old[2:,1:-1] + u_old[:-2,1:-1])*dy2  \
                    + (u_old[1:-1,2:] + u_old[1:-1,:-2])*dx2) \
                 / (2*(dx2 + dy2))

def calc(N,Niter=100,func=naive_update):
    dx = 1./(N - 1)
    dy = 1./(N - 1)
    u = np.zeros([N,N])
    u[0,:] = 1.
    u[:,0] = 1.
    for i in range(Niter):
        func(u,dx,dy)
    return u

if __name__ == "__main__":

    # naive update
    t_start = timer()
    u1 = calc(N=200,Niter=100,func=naive_update)
    t_end   = timer()
    print("naive_update: time = {:8.5f} seconds".format(t_end - t_start))

    # numpy update
    t_start = timer()
    u2= calc(N=200,Niter=100,func=numpy_update)
    t_end   = timer()
    print("numpy_update: time = {:8.5f} seconds".format(t_end - t_start))

