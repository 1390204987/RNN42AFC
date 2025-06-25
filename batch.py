# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:24:14 2023

@author: NaN
"""
import numpy as np
from mytrain2 import *

#for search the proper parameter of rnn1 recurence, rnn2 recurence, rnn1 noise, rnn2 noise
recur1 = np.linspace(0.1,1,num=5)
recur2 = np.linspace(0.1,1,num=5)
# noise1 = np.linspace(0.1,1,num=10)
# noise2 = np.linspace(0.1,1,num=10)
# feedforward_stren = np.linspace(0.1,1,num=10)
# feedback_stren = np.linspace(0,1,num=10)
# idnet = 5800
for i in range(len(recur1)):
    i_recur1 = recur1[i]

    for ii in range(len(recur2)):
        ii_recur2 = recur2[ii]
        # ii_recur2 = 0.5

        for iii in [0,1,2,3]:
            # iii_ff_stren = feedforward_stren[iii]            
            for iiii in [0,1,2,3]:
                # iiii_fb_stren = feedback_stren[iiii]     
                idnet = i*1000+ii*100+iii*10+iiii
                random_num = iii*10+iiii                               
                hp = get_default_hp('coltargdm')                
                hp["recur1"] = i_recur1
                hp["recur2"] = ii_recur2 
                hp["fforwardstren"] = 1
                hp["fbackstren"] = 1   
                # hp['sigma_rec1'] = noise1[i]
                # hp['sigma_rec2'] = noise2[ii]
                hp["seed"] = random_num
                formatted_idnet = str(idnet).zfill(4)
                netname = formatted_idnet+'colorhdnet4'     
                savepath = './checkpoint_batchnew1/'
                train(netname,savepath,hp=hp, ruleset = 'coltargdm')
                # idnet = idnet+1