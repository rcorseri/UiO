#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:29:22 2022

@author: rpcorser
"""
import numpy as np


def DesignMatrix(x, y, n):
    x = x.ravel()
    y = y.ravel()
    length = len(x)
    if n == 1:
        X =np.stack((np.ones(length), x , y), axis=-1)
    elif n == 2:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y), axis=-1)
    elif n == 3:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**1), (x**2)*(y**1)), axis=-1)
    elif n == 4:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**1), (x**2)*(y**1), (x**2)*(y**2),(x**1)*(y**3), (x**1)*(y**3), x**4, y**4), axis=-1)
    elif n == 5:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**1), (x**2)*(y**1), (x**2)*(y**2),(x**1)*(y**3), (x**1)*(y**3), x**4, y**4, (x**4)*(y**1),(x**1)*(y**4),(x**3)*(y**2), (x**2)*(y**3), x**5, y**5), axis=-1)
    elif n == 6:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**1), (x**2)*(y**1), (x**2)*(y**2),(x**1)*(y**3), (x**1)*(y**3), x**4, y**4, (x**4)*(y**1),(x**1)*(y**4),(x**3)*(y**2), (x**2)*(y**3), x**5, y**5,(x**5)*(y**1),(x**1)*(y**5), (x**4)*(y**2),(x**2)*(y**4),(x**3)*(y**3), x**6, y**6),axis=-1)
    elif n == 7:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**1), (x**2)*(y**1), (x**2)*(y**2),(x**1)*(y**3), (x**1)*(y**3), x**4, y**4, (x**4)*(y**1),(x**1)*(y**4),(x**3)*(y**2), (x**2)*(y**3), x**5, y**5,(x**5)*(y**1),(x**1)*(y**5), (x**4)*(y**2),(x**2)*(y**4),(x**3)*(y**3), x**6, y**6, (x**1)*(y**6),(x**6)*(y**1),(x**2)*(y**5),(x**5)*(y**2),(x**3)*(y**4),(x**4)*(y**3),y**7, y**7),axis=-1)
    return X
