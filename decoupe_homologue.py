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
#Lecture images puis création de la classe homologue 
##############################################################################

def decoupe_homologue(taille_ligne, taille_colonne):  
    """
    taille_ligne : correspond au nombre de pixel en vertical
    taille_colonne : correspond au nombre de pixel en horizontal
    
    Enregistre les images des descripteurs de points homologues de 
    taille (taille_ligne, taille_colonne) dans le dossier où se trouve le script
    """
    l = 1
    path = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_1/eTPR_GrayMax/eTVIR_ACGT/*.tif') 
    for pic in glob.glob(path):  
        im = imageio.imread(pic) 
        for k in range (int(len(im)/taille_ligne)):
            new_im = np.zeros((taille_ligne, taille_colonne))
            for i in range(taille_ligne):
                for j in range (taille_colonne):
                    new_im[i][j]=im[i+k*taille_ligne][j] 
            imageio.imsave('%s.jpg'%l, new_im)
            l+=1
    return

if __name__ == "__main__":
    # images 10*20 pixels: descripteurs ACR0 et ACGT
    print(decoupe_homologue(10,20))
    
    # images 10*18 pixels: descripteur ACGR
    #print(decoupe_homologue(10,18))
    
    
    