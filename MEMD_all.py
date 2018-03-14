# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:50:30 2018

@author: Mario de Souza e Silva


This is a translation of the MEMD (Multivariate Empirical Mode Decomposition)
code from Matlab to Python.

The Matlab code was developed by [1] and is freely available at:
    . http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm

The only difference in this Python script is that the input data can have any
number of channels, instead of the 36 estabilished in the original Matlab code.

All of the defined functions have been joined together in one single script
called MEMD_all. Bellow follows the Syntax described in [1], but adapted to
this Python script.

-------------------------------------------------------------------------------
Syntax:
from MEMD_all import memd

imf = memd(X)
  returns a 3D matrix 'imf(M,N,L)' containing M multivariate IMFs, one IMF per column, computed by applying
  the multivariate EMD algorithm on the N-variate signal (time-series) X of length L.
   - For instance, imf_k = IMF[:,k,:] returns the k-th component (1 <= k <= N) for all of the N-variate IMFs.

  For example,  for hexavariate inputs (N=6), we obtain a 3D matrix IMF(M, 6, L)
  where M is the number of IMFs extracted, and L is the data length.

imf = memd(X,num_directions)
  where integer variable num_directions (>= 1) specifies the total number of projections of the signal
    - As a rule of thumb, the minimum value of num_directions should be twice the number of data channels,
    - for instance, num_directions = 6  for a 3-variate signal and num_directions= 16 for an 8-variate signal
  The default number of directions is chosen to be 64 - to extract meaningful IMFs, the number of directions
  should be considerably greater than the dimensionality of the signals

imf = memd(X,num_directions,'stopping criteria')
  uses the optional parameter 'stopping criteria' to control the sifting process.
   The available options are
     -  'stop' which uses the standard stopping criterion specified in [2]
     -  'fix_h' which uses the modified version of the stopping criteria specified in [3]
   The default value for the 'stopping criteria' is 'stop'.

 The settings  num_directions=64 and 'stopping criteria' = 'stop' are defaults.
    Thus imf = memd(X) = memd(X,64) = memd(X,64,'stop') = memd(X,None,'stop'),

imf = memd(X, num_directions, 'stop', stop_vec)
  computes the IMFs based on the standard stopping criterion whose parameters are given in the 'stop_vec'
    - stop_vec has three elements specifying the threshold and tolerance values used, see [2].
    - the default value for the stopping vector is   step_vec = (0.075,0.75,0.075).
    - the option 'stop_vec' is only valid if the parameter 'stopping criteria' is set to 'stop'.

imf = memd(X, num_directions, 'fix_h', n_iter)
  computes the IMFs with n_iter (integer variable) specifying the number of consecutive iterations when
  the number of extrema and the number of zero crossings differ at most by one [3].
    - the default value for the parameter n_iter is set to  n_iter = 2.
    - the option n_iter is only valid if the parameter  'stopping criteria' = 'fix_h'


This code allows to process multivaraite signals having any number of channels, using the multivariate EMD algorithm [1].
  - to process 1- and 2-dimensional (univariate and bivariate) data using EMD in Python, it is recommended the toolbox from
                https://bitbucket.org/luukko/libeemd 

Acknowledgment: All of this code is based on the multivariate EMD code, publicly available from
                http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm.


[1]  Rehman and D. P. Mandic, "Multivariate Empirical Mode Decomposition", Proceedings of the Royal Society A, 2010
[2]  G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode Decomposition and its Algorithms", Proc of the IEEE-EURASIP
     Workshop on Nonlinear Signal and Image Processing, NSIP-03, Grado (I), June 2003
[3]  N. E. Huang et al., "A confidence limit for the Empirical Mode Decomposition and Hilbert spectral analysis",
     Proceedings of the Royal Society A, Vol. 459, pp. 2317-2345, 2003


Usage 


Case 1:
import numpy as np

np.random.rand(100,3)
imf = memd(inp)
imf_x = imf[:,0,:] #imfs corresponding to 1st component
imf_y = imf[:,1,:] #imfs corresponding to 2nd component
imf_z = imf[:,2,:] #imfs corresponding to 3rd component


Case 2:
import numpy as np

inp = np.loadtxt('T045.txt')
imf = memd(inp,256,'stop',(0.05,0.5,0.05))

"""

import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from math import pi,sqrt,sin,cos
import warnings
import sys



# =============================================================================

def hamm(n,base):
    seq = np.zeros((1,n))
    
    if 1 < base:
        seed = np.arange(1,n+1)
        base_inv = 1/base
        while any(x!=0 for x in seed):
            digit = np.remainder(seed[0:n],base)
            seq = seq + digit*base_inv
            base_inv = base_inv/base
            seed = np.floor (seed/base)
    else:
        temp = np.arange(1,n+1)
        seq = (np.remainder(temp,(-base+1))+0.5)/(-base)
        
    return(seq)

# =============================================================================

def zero_crossings(x):
    indzer = np.where(x[0:-1]*x[1:]<0)[0]
    
    if any(x == 0):
        iz = np.where(x==0)[0]
        if any(np.diff(iz)==1):
            zer = x == 0
            dz = np.diff([0,zer,0])
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0]-1
            indz = np.round((debz+finz)/2)
        else:
            indz = iz
        indzer = np.sort(np.concatenate((indzer,indz)))
        
    return(indzer)

# =============================================================================

#defines new extrema points to extend the interpolations at the edges of the
#signal (mainly mirror symmetry)
def boundary_conditions(indmin,indmax,t,x,z,nbsym):
    lx = len(x)-1
    end_max = len(indmax)-1
    end_min = len(indmin)-1
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    if len(indmin) + len(indmax) < 3:
        mode = 0
        tmin=tmax=zmin=zmax=None
        return(tmin,tmax,zmin,zmax,mode)
    else:
        mode=1 #the projected signal has inadequate extrema
    #boundary conditions for interpolations :
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1:min(end_max+1,nbsym+1)])
            lmin = np.flipud(indmin[:min(end_min+1,nbsym)])
            lsym = indmax[0]

        else:
            lmax = np.flipud(indmax[:min(end_max+1,nbsym)])
            lmin = np.concatenate((np.flipud(indmin[:min(end_min+1,nbsym-1)]),([0])))
            lsym = 0

    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[:min(end_max+1,nbsym)])
            lmin = np.flipud(indmin[1:min(end_min+1,nbsym+1)])
            lsym = indmin[0]

        else:
            lmax = np.concatenate((np.flipud(indmax[:min(end_max+1,nbsym-1)]),([0])))
            lmin = np.flipud(indmin[:min(end_min+1,nbsym)])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max-nbsym+1,0):])
            rmin = np.flipud(indmin[max(end_min-nbsym,0):-1])
            rsym = indmin[-1]

        else:
            rmax = np.concatenate((np.array([lx]),np.flipud(indmax[max(end_max-nbsym+2,0):])))
            rmin = np.flipud(indmin[max(end_min-nbsym+1,0):])
            rsym = lx

    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max-nbsym,0):-1])
            rmin = np.flipud(indmin[max(end_min-nbsym+1,0):])
            rsym = indmax[-1]

        else:
            rmax = np.flipud(indmax[max(end_max-nbsym+1,0):])
            rmin = np.concatenate((np.array([lx]),np.flipud(indmin[max(end_min-nbsym+2,0):])))
            rsym = lx

    tlmin = 2*t[lsym]-t[lmin]
    tlmax = 2*t[lsym]-t[lmax]
    trmin = 2*t[rsym]-t[rmin]
    trmax = 2*t[rsym]-t[rmax]

    #in case symmetrized parts do not extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:min(end_max+1,nbsym)])
        else:
            lmin = np.flipud(indmin[:min(end_min+1,nbsym)])
        if lsym == 1:
            sys.exit('bug')
        lsym = 0
        tlmin = 2*t[lsym]-t[lmin]
        tlmax = 2*t[lsym]-t[lmax]
        
    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max-nbsym+1,0):])
        else:
            rmin = np.flipud(indmin[max(end_min-nbsym+1,0):])
        if rsym == lx:
            sys.exit('bug')
        rsym = lx
        trmin = 2*t[rsym]-t[rmin]
        trmax = 2*t[rsym]-t[rmax]

    zlmax =z[lmax,:]
    zlmin =z[lmin,:]
    zrmax =z[rmax,:]
    zrmin =z[rmin,:]

    tmin = np.hstack((tlmin,t[indmin],trmin))
    tmax = np.hstack((tlmax,t[indmax],trmax))
    zmin = np.vstack((zlmin,z[indmin,:],zrmin))
    zmax = np.vstack((zlmax,z[indmax,:],zrmax))

    return(tmin,tmax,zmin,zmax,mode)

# =============================================================================

# computes the mean of the envelopes and the mode amplitude estimate
def envelope_mean(m,t,seq,ndir,N,N_dim): #new

    NBSYM = 2
    count = 0

    env_mean=np.zeros((len(t),N_dim))
    amp = np.zeros((len(t)))
    nem = np.zeros((ndir))
    nzm = np.zeros((ndir))
    
    dir_vec = np.zeros((N_dim,1))
    for it in range(0,ndir):
        if N_dim !=3:     # Multivariate signal (for N_dim ~=3) with hammersley sequence
            #Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
            b=2*seq[it,:]-1 
            
            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1]**2)))\
                             ,b[:N_dim-1]).transpose()
            
            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:,0] = np.cumprod(np.concatenate(([1],np.sin(tht))))
            dir_vec[:N_dim-1,0] =  np.cos(tht)*dir_vec[:N_dim-1,0]
            
        else:     # Trivariate signal with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
            tt = 2*seq[it,0]-1
            if tt>1:
                tt=1
            elif tt<-1:
                tt=-1         
            
            # Normalize angle from 0 - 2*pi
            phirad = seq[it,1]*2*pi
            st = sqrt(1.0-tt*tt)
            
            dir_vec[0]=st*cos(phirad)
            dir_vec[1]=st*sin(phirad)
            dir_vec[2]=tt
           
        # Projection of input signal on nth (out of total ndir) direction vectors
        y  = np.dot(m,dir_vec)

        # Calculates the extrema of the projected signal
        indmin,indmax = local_peaks(y)      

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin,tmax,zmin,zmax,mode = boundary_conditions(indmin,indmax,t,y,m,NBSYM)
        
        # Calculate multidimensional envelopes using spline interpolation
        # Only done if number of extrema of the projected signal exceed 3
        if mode:
            fmin = CubicSpline(tmin,zmin,bc_type='not-a-knot')
            env_min = fmin(t)
            fmax = CubicSpline(tmax,zmax,bc_type='not-a-knot')
            env_max = fmax(t)
            amp = amp + np.sqrt(np.sum(np.power(env_max-env_min,2),axis=1))/2
            env_mean = env_mean + (env_max+env_min)/2
        else:     # if the projected signal has inadequate extrema
            count=count+1
            
    if ndir>count:
        env_mean = env_mean/(ndir-count)
        amp = amp/(ndir-count)
    else:
        env_mean = np.zeros((N,N_dim))
        amp = np.zeros((N))
        nem = np.zeros((ndir))
        
    return(env_mean,nem,nzm,amp)

# =============================================================================

#Stopping criterion
def stop(m,t,sd,sd2,tol,seq,ndir,N,N_dim):
    try:
        env_mean,nem,nzm,amp = envelope_mean(m,t,seq,ndir,N,N_dim)
        sx = np.sqrt(np.sum(np.power(env_mean,2),axis=1))
        
        if all(amp):     # something is wrong here
            sx = sx/amp
            
        if ((np.mean(sx > sd) > tol or any(sx > sd2)) and any(nem > 2)) == False:
            stp = 1
        else:
            stp = 0
    except:
        env_mean = np.zeros((N,N_dim))
        stp = 1
        
    return(stp,env_mean)
    
# =============================================================================
    
def fix(m,t,seq,ndir,stp_cnt,counter,N,N_dim):
    try:
        env_mean,nem,nzm,amp = envelope_mean(m,t,seq,ndir,N,N_dim)
        
        if all(np.abs(nzm-nem)>1):
            stp = 0
            counter = 0
        else:
            counter = counter+1
            stp = (counter >= stp_cnt)
    except:
        env_mean = np.zeros((N,N_dim))
        stp = 1
        
    return(stp,env_mean,counter)

# =============================================================================

def peaks(X):
    dX = np.sign(np.diff(X.transpose())).transpose()
    locs_max = np.where(np.logical_and(dX[:-1] >0,dX[1:] <0))[0]+1
    pks_max = X[locs_max]
    
    return(pks_max,locs_max)

# =============================================================================

def local_peaks(x):
    if all(x < 1e-5):
        x=np.zeros((1,len(x)))

    m = len(x)-1
    
    # Calculates the extrema of the projected signal
    # Difference between subsequent elements:
    dy = np.diff(x.transpose()).transpose()
    a = np.where(dy!=0)[0]
    lm = np.where(np.diff(a)!=1)[0] + 1
    d = a[lm] - a[lm-1] 
    a[lm] = a[lm] - np.floor(d/2)
    a = np.insert(a,len(a),m)
    ya  = x[a]
    
    if len(ya) > 1:
        # Maxima
        pks_max,loc_max=peaks(ya)
        # Minima
        pks_min,loc_min=peaks(-ya)
        
        if len(pks_min)>0:
            indmin = a[loc_min]
        else:
            indmin = np.asarray([])
            
        if len(pks_max)>0:
            indmax = a[loc_max]
        else:
            indmax = np.asarray([])
    else:
        indmin=np.array([])
        indmax=np.array([])
        
    return(indmin, indmax)

# =============================================================================

def stop_emd(r,seq,ndir,N_dim):
    ner = np.zeros((ndir,1))
    dir_vec = np.zeros((N_dim,1))
    
    for it in range(0,ndir):
        if N_dim != 3: # Multivariate signal (for N_dim ~=3) with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
            b=2*seq[it,:]-1
            
            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1]**2)))\
                             ,b[:N_dim-1]).transpose()
            
            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:,0] = np.cumprod(np.concatenate(([1],np.sin(tht))))
            dir_vec[:N_dim-1,0] =  np.cos(tht)*dir_vec[:N_dim-1,0]
    
        else: # Trivariate signal with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
            tt = 2*seq[it,0]-1
            if tt>1:
                tt=1
            elif tt<-1:
                tt=-1  
            
            # Normalize angle from 0 - 2*pi
            phirad = seq[it,1]*2*pi
            st = sqrt(1.0-tt*tt)
            
            dir_vec[0]=st*cos(phirad)
            dir_vec[1]=st*sin(phirad)
            dir_vec[2]=tt
        # Projection of input signal on nth (out of total ndir) direction
        # vectors
        y = np.dot(r,dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        ner[it] = len(indmin) + len(indmax)
    
    # Stops if the all projected signals have less than 3 extrema
    stp = all(ner<3)
    
    return (stp)

# =============================================================================

def is_prime(x):
    if x == 2:
        return True
    else:
        for number in range (3,x): 
            if x % number == 0 or x % 2 == 0:
         #print number
                return (False)
            
# =============================================================================
                
        return (True)
def nth_prime(n):
    lst = [2]
    for i in range(3,104745):
        if is_prime(i) == True:
            lst.append(i)
            if len(lst) == n:
                return (lst)
# =============================================================================

def set_value(*args):
    args = args[0]
    narg = len(args)
    q = args[0]
    
    ndir=stp_cnt=MAXITERATIONS=sd=sd2=tol = None
    stp_crit,stp_vec,base = [],[],[]
                                          
    if narg == 0:
        sys.exit('Not enough input arguments.')
    elif narg > 4:
        sys.exit('Too many input arguments.')
    elif narg == 1:
        ndir = 64     # default
        stp_crit = 'stop'     # default
        stp_vec = np.array([0.075,0.75,0.075])     # default
        sd,sd2,tol = stp_vec[0],stp_vec[1],stp_vec[2]        
    elif narg == 2:
        ndir = args[1]
        stp_crit = 'stop'     # default
        stp_vec = np.array([0.075,0.75,0.075])     # default
        sd,sd2,tol = stp_vec[0],stp_vec[1],stp_vec[2]
    elif narg == 3:
        if args[1] != None:
            ndir = args[1]
        else:
            ndir = 64     # default
        stp_crit = args[2]
        if stp_crit == 'stop':
            stp_vec = np.array([0.075,0.75,0.075])     # default
            sd,sd2,tol = stp_vec[0],stp_vec[1],stp_vec[2]
        elif stp_crit == 'fix_h':
            stp_cnt = 2     # default
    elif narg == 4:
        if args[1] != None:
            ndir = args[1]
        else:
            ndir = 64     # default        
        stp_crit = args[2]        
        if args[2] == 'stop':
            stp_vec = args[3]
            sd,sd2,tol = stp_vec[0],stp_vec[1],stp_vec[2]
        elif args[2] == 'fix_h':
            stp_cnt = args[3]

    # Rescale input signal if required
    if len(q) == 0:                                                            # Doesn't do the same as the Matlab script
        sys.exit('emptyDataSet. Data set cannot be empty.')
    if np.shape(q)[0] < np.shape(q)[1]:
        q=q.transpose()
        
    # Dimension of input signal
    N_dim = np.shape(q)[1]
    if N_dim < 3:
        sys.exit('Function only processes the signal having more than 3.')
        
    # Length of input signal
    N = np.shape(q)[0]

    # Check validity of Input parameters                                       #  Doesn't do the same as the Matlab script
    if not isinstance(ndir,int) or ndir < 6:
        sys.exit('invalid num_dir. num_dir should be an integer greater than or equal to 6.')
    if not isinstance(stp_crit, str) or (stp_crit != 'stop' and stp_crit != 'fix_h'):
        sys.exit('invalid stop_criteria. stop_criteria should be either fix_h or stop')
    if not isinstance(stp_vec,(list, tuple, np.ndarray)) or any(x for x in stp_vec if not isinstance(x,(int, float, complex))):
        sys.exit('invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]')
    if stp_cnt != None:
        if not isinstance(stp_cnt,int) or stp_cnt < 0:
            sys.exit('invalid stop_count. stop_count should be a nonnegative integer.')

    # Initializations for Hammersley function
    base.append(-ndir)
    
    # Find the pointset for the given input signal
    if N_dim==3:
        base.append(2)
        seq = np.zeros((ndir,N_dim-1))
        for it in range(0,N_dim-1):
            seq[:,it] = hamm(ndir,base[it])
    else:
        #Prime numbers for Hammersley sequence
        prm = nth_prime(N_dim-1)
        for itr in range(1,N_dim):
            base.append(prm[itr-1])
        seq = np.zeros((ndir,N_dim))
        for it in range(0,N_dim):
            seq[:,it] = hamm(ndir,base[it])
    # Define t
    t = np.arange(1,N+1)
    #Counter
    nbit = 0
    MAXITERATIONS = 1000     #default    
    
    return(q,seq,t,ndir,N_dim,N,sd,sd2,tol,nbit,MAXITERATIONS,stp_crit,stp_cnt)
    
# =============================================================================
    
def memd(*args):
    x,seq,t,ndir,N_dim,N,sd,sd2,tol,nbit,MAXITERATIONS,stop_crit,stp_cnt = set_value(args)

    r=x
    n_imf=1
    q = []

    while stop_emd(r,seq,ndir,N_dim) == False:
        # current mode
        m = r
        
        # computation of mean and stopping criterion
        if stop_crit == 'stop':
            stop_sift,env_mean = stop(m,t,sd,sd2,tol,seq,ndir,N,N_dim)
        else:
            counter=0
            stop_sift,env_mean,counter = fix(m,t,seq,ndir,stp_cnt,counter,N,N_dim)
            
        # In case the current mode is so small that machine precision can cause
        # spurious extrema to appear
        if np.max(np.abs(m)) < (1e-10)*(np.max(np.abs(x))):
            if stop_sift == False:
                warnings.warn('emd:warning','forced stop of EMD : too small amplitude')
            else:
                print('forced stop of EMD : too small amplitude')
            break
        
        # sifting loop
        while stop_sift == False and nbit < MAXITERATIONS:
            # sifting
            m = m - env_mean
            
            # computation of mean and stopping criterion
            if stop_crit =='stop':
                stop_sift,env_mean = stop(m,t,sd,sd2,tol,seq,ndir,N,N_dim)
            else:
                stop_sift,env_mean,counter = fix(m,t,seq,ndir,stp_cnt,counter,N,N_dim)
        
            nbit=nbit+1
            
            if nbit == (MAXITERATIONS-1) and  nbit > 100:
                warnings.wanr('emd:warning','forced stop of sifting : too many erations')
            
        q.append(m.transpose())
        
        n_imf = n_imf+1
        r = r - m
        nbit = 0
        
    # Stores the residue
    q.append(r.transpose())
    q = np.asarray(q)
    #sprintf('Elapsed time: %f\n',toc);

    return(q)
# =============================================================================