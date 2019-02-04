#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:29:26 2018

@author: guillemettefonteix
"""

##############################################################################
#Import des librairies
##############################################################################

import numpy as np
import os
import glob
import imageio

##############################################################################
#Lecture images puis création de la classe homologue (images 10*20 pixels)
##############################################################################

def decoupe_homologue():  
    l = 1
    path = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_1/eTPR_BifurqMax/') 
    for pic in glob.glob(path+'/eTVIR_ACR0/*.tif'):  
        im = imageio.imread(pic) 
        for k in range (int(len(im)/10)):
            new_im = np.zeros((10, 20))
            for i in range(10):
                for j in range (20):
                    new_im[i][j]=im[i+k*10][j] 
            imageio.imsave('%s.jpg'%l, new_im)
            l+=1
    return

if __name__ == "__main__":
    print(decoupe_homologue())