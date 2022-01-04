import numpy as np
from scipy.special import jv, yv
import torch
import time
from multiprocessing import Pool
import os
def product(a,b):
    z = np.array([a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3]]).transpose()
    return z

def besselj(m,x):
    if np.isnan(np.sqrt(x)).any():
        print("in besselj: your x is nan")
        print(x)
    return jv(m+0.5,x)/np.sqrt(x)

def besseljd(m,x):
    return (m*besselj(m-1,x) - (m+1)*besselj(m+1,x))/(2*m+1)

def bessely(m,x):
    return yv(m+0.5,x)/np.sqrt(x)

def besselyd(m,x):
    return (m*bessely(m-1,x) - (m+1)*bessely(m+1,x))/(2*m+1)

def spherical_TM1(k,l,r_cumel,omega,eps1,eps2):
    k1 = omega*np.sqrt(eps1)
    k2 = omega*np.sqrt(eps2)
    x1 = k1*r_cumel
    x2 = k2*r_cumel

    j1 = besselj(l,x1)
    j1d = besseljd(l,x1)*x1 + j1
    y1 = bessely(l,x1)
    y1d = besselyd(l,x1)*x1 + y1
    j2 = besselj(l,x2)
    j2d = besseljd(l,x2)*x2 + j2
    y2 = bessely(l,x2)
    y2d = besselyd(l,x2)*x2 + y2

    if k == 1: #TE Mode
        M = product([y2d,-j2d,-y2,j2],[j1,j1d,y1,y1d])
    else: #TM Mode
        M = product([eps1*y2d,-eps1*j2d,-y2,j2], [j1,eps2*j1d,y1,eps2*y1d])

    M = [M[:,0], M[:,1], M[:,2], M[:,3]]
    return M

def spherical_TM2(k,l,r,omega,eps):
    cum_r = np.cumsum(r)
    N,K = eps.shape
    K = K - 1
    M = np.empty((N,4))
    M[:,0] = 1
    M[:,1:2] = 0
    M[:,3] = 1
    M = [M[:,0],M[:,1],M[:,2],M[:,3]]

    for i in range(0,K):
        tmp = spherical_TM1(k,l,cum_r[i],omega,eps[:,i],eps[:,i+1])
        tmp = [tmp[0], tmp[1], tmp[2], tmp[3]]
        M = product(tmp,M)
        M = [M[:,0], M[:,1], M[:,2], M[:,3]]
        # product is giving some strange output here, the resulting M is (4,4), but it should be (401,4)

    return M

def spherical_cs(k,l,r,omega,eps):
    M = spherical_TM2(k,l,r,omega,eps)
    tmp = M[0]/M[1]

    R = (tmp - 1j)/(tmp + 1j)
    R = np.expand_dims(R,axis=1)

    coef = (2*l+1)*np.pi/2*(1/np.power(omega,2))*(1/eps[:,-1])
    coef = np.expand_dims(coef,axis=1)

    z = 1-np.power(np.abs(R),2)
    y = np.power(np.abs(1-R),2)

    sigma = np.concatenate((coef,coef),axis=1)*np.concatenate((z,y),axis=1)

    return sigma

def total_cs(r,omega,eps,order):
    sigma = 0
    for o in range(1,order+1):
        sigma = sigma + spherical_cs(1,o,r,omega,eps) + spherical_cs(2,o,r,omega,eps)
    return sigma

def simulate(radii,lamLimit=400,orderLimit=None,epsIn=None):
    if not epsIn:
        lam = np.linspace(lamLimit, 800, (800-lamLimit)+1)
        omega = 2*np.pi/lam

        eps_silica = 2.04 * np.ones(len(omega))
        my_lam = lam/1000
        eps_tio2 = 5.913+(.2441) * 1/(my_lam*my_lam - .0803)
        eps_water = 1.77 * np.ones(len(omega))

        eps = np.empty((len(lam),len(radii)+1))

        for idx in range(len(radii)):
            if idx%2 == 0:
                eps[:,idx] = eps_silica
            else:
                eps[:,idx] = eps_tio2

        eps[:,-1] = eps_water

    else:
        eps = epsIn

    order = 25
    if len(radii) == 2 or len(radii) == 3:
        order = 4
    elif len(radii) == 4 or len(radii) == 5:
        order = 9
    elif len(radii) == 6 or len(radii) == 7:
        order = 12
    elif len(radii) == 8 or len(radii) == 9:
        order = 15
    elif len(radii) == 10 or len(radii) == 11:
        order = 18

    if None != orderLimit:
        order = orderLimit

    spect = total_cs(radii,omega,eps,order)/(np.pi*np.power(np.sum(radii),2))
    processed_spect = spect[0::2,1]

    return processed_spect



def generate(low_bound,up_bound,num_samples,num_layers, rand_seed=42):
    data_y = []
    data_x = []
    # Set the random seed
    np.random.seed(rand_seed)
    for n in range(num_samples):
        r = np.round(np.random.rand(num_layers) * (up_bound - low_bound) + low_bound, 1)
        #r = rad[n]

        spect = simulate(r)
        data_x.append(r)
        data_y.append(spect)

    # Convert list to np array
    data_x = np.array(data_x)
    data_y = np.array(data_y)
        # Computationally expensive for concatenate
        # r = np.expand_dims(r,axis=0)
        # dz = np.expand_dims(spect,axis=0)

        # if len(data_x) == 0:
        #     data_x = r
        #     data_y = dz
        # else:
        #     data_x = np.concatenate((data_x,r))
        #     data_y = np.concatenate((data_y,dz))

    return data_x, data_y

def generate_multi_processing(index, num_samples):
    """
    The helper function to be called using multi-processing generation
    index: The i-th batch of things
    num_samples: The number of samples to be generated in this batch of run
    """
    low_bound = 30
    up_bound = 70
    num_layers = 8

    data_x, data_y = generate(low_bound,up_bound,num_samples,num_layers, rand_seed=index)
    np.savetxt('data_x_{}.csv'.format(index),data_x,delimiter=',')
    np.savetxt('data_y_{}.csv'.format(index),data_y,delimiter=',')
    return data_x, data_y

def combine_multi_processing(num_files):
    """
    Re combine the parallelly generated data into a single file
    They have duplicated entry!!!
    """
    data_x_full, data_y_full = None, None
    for i in range(num_files):
        # Read csv
        data_x = np.loadtxt('data_x_{}.csv'.format(i),delimiter=',')
        data_y = np.loadtxt('data_y_{}.csv'.format(i),delimiter=',')
        # Append them
        if data_x_full is None:
            data_x_full = np.zeros([num_files, *np.shape(data_x)])
            data_y_full = np.zeros([num_files, *np.shape(data_y)])
        data_x_full[i, :, :] = data_x
        data_y_full[i, :, :] = data_y

        # Delete those csv
        # os.remove('data_x_{}.csv'.format(i))
        # os.remove('data_y_{}.csv'.format(i))
    
    # Reshape into a large, new array
    data_x_full = np.reshape(data_x_full, [-1, np.shape(data_x_full)[-1]])
    data_y_full = np.reshape(data_y_full, [-1, np.shape(data_y_full)[-1]])
    print(np.shape(data_x_full))
    print(np.shape(data_y_full))
    np.savetxt('data_x.csv', data_x_full, delimiter=',')
    np.savetxt('data_y.csv', data_y_full, delimiter=',')



if __name__ == '__main__':
    # # Normal bound
    start = time.time()
    num_cpu = 10
    ndata = 50000   # Training and validation set
    # ndata = 1000    # Test set (half would be taken)
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((i, ndata//num_cpu))
        # print((args_list))
        # print(len(args_list))
        X_list = pool.starmap(generate_multi_processing, args_list)
    finally:
        pool.close()
        pool.join()
    print(time.time()-start)
    print(len(X_list))
    print(len(X_list[0]))
    print(X_list[0][0])
    print(X_list[1][0])

    combine_multi_processing(num_cpu)


    """
    # Extended bound
    start = time.time()
    low_bound = 30
    up_bound = 90
    num_layers = 3
    num_samples = 2000

    data_x, data_y = generate(low_bound,up_bound,num_samples,num_layers)
    np.savetxt("data_x_extended.csv",data_x,delimiter=',')
    np.savetxt("data_y_extended.csv",data_y,delimiter=',')
    print(time.time()-start)
    """
