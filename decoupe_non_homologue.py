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
#Lecture image puis création de la classe non homologue 
#############################################################################

def decoupe_non_homologue(taille_ligne, taille_colonne):  
    """
    taille_ligne : correspond au nombre de pixel en vertical
    taille_colonne : correspond au nombre de pixel en horizontal
    
    Enregistre les images des descripteurs de points non-homologues
    de taille (taille_ligne, taille_colonne) dans le dossier où se trouve le script
    """
    l = 1
    path = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_1/eTPR_GrayMax/eTVIR_ACGT/*.tif') 
    
    # On parcourt toutes les images tif du dossier 
    for pic in glob.glob(path):  
        im = imageio.imread(pic)  # Lecture de l'image .tif
        n = int(len(im)/taille_ligne)-1
        m = int(taille_colonne/2)
        for k in range (n): 
            new_im = np.zeros((taille_ligne, taille_colonne))   #création d'une matrice de taille 10*20 qui correspondra à notre couple de descripteurs
            r = random.randint(1,n)       #On prend un nombre au hasard (pour apparier des descripteurs non-homologues)
            
            for i in range(taille_ligne):               
                for j in range (m):
                    new_im[i][j]=im[i+k*taille_ligne][j]             
                    if (k!=r):
                        new_im[i][j+m]=im[i+r*taille_ligne][j+m] 
                    else:
                        new_im[i][j+m]=im[i+(r-1)*taille_ligne][j+m] 
            imageio.imsave('%sn.jpg'%l, new_im)
            l+=1
    return

if __name__ == "__main__":
    #images 10x20 pixels: descripteurs ACR0 et ACGT
    print(decoupe_non_homologue(10, 20))
    
    # images 10*18 pixels: descripteur ACGR
    #print(decoupe_non_homologue(10, 18))