# MEMD-Python-
Python version of the Multivariate Empirical Mode Decomposition algorith


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
