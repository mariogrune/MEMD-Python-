# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:43:59 2018

@author: mario
"""

import numpy as np

from MEMD_all import memd

inp = np.loadtxt('T045.txt')
imf = memd(inp)
imf_x = imf[:,0,:]
imf_y = imf[:,1,:]
imf_z = imf[:,2,:]
