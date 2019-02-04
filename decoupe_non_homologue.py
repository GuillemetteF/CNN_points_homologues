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
import random

#############################################################################
#Lecture image puis création de la classe non homologue (image 10x20 pixels)
#############################################################################

def decoupe_non_homologue():    
    l = 1
    path = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_1/eTPR_BifurqMax/') 
    
    # On parcourt toutes les images tif du dossier 
    for pic in glob.glob(path+'/eTVIR_ACR0/*.tif'):  
        im = imageio.imread(pic)  # Lecture de l'image .tif
        n = int(len(im)/10)-1
        for k in range (n): 
            new_im = np.zeros((10, 20))   #création d'une matrice de taille 10*20 qui correspondra à notre couple de descripteurs
            r = random.randint(1,n)       #On prend un nombre au hasard (pour apparier des descripteurs non-homologues)
            
            for i in range(10):               
                for j in range (10):
                    new_im[i][j]=im[i+k*10][j]             
                    if (k!=r):
                        new_im[i][j+10]=im[i+r*10][j+10] 
                    else:
                        new_im[i][j+10]=im[i+(r-1)*10][j+10] 
            imageio.imsave('%sn.jpg'%l, new_im)
            l+=1
    return

if __name__ == "__main__":
    print(decoupe_non_homologue())