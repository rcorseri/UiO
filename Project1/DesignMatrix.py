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
    if   n == 0:
        X =np.stack((np.ones(length)), axis=-1)
    elif n == 1:
        X =np.stack((np.ones(length), x , y), axis=-1)
    elif n == 2:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y), axis=-1)
    elif n == 3:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2)), axis=-1)
    elif n == 4:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3)), axis=-1)
    elif n == 5:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4)), axis=-1)
    elif n == 6:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4),x**6, y**6,(x**5)*(y**5)), axis=-1)
    elif n == 7:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4),x**6, y**6,(x**5)*(y**5),x**7, y**7,(x**6)*(y**6)), axis=-1)
    elif n == 8:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4),x**6, y**6,(x**5)*(y**5),x**7, y**7,(x**6)*(y**6),x**8, y**8,(x**7)*(y**7)), axis=-1)
    elif n == 9:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4),x**6, y**6,(x**5)*(y**5),x**7, y**7,(x**6)*(y**6),x**8, y**8,(x**7)*(y**7),x**9, y**9,(x**8)*(y**8)), axis=-1)
    elif n == 10:
        X =np.stack((np.ones(length), x , y , x**2 , y**2 , x*y , x**3 , y**3 ,(x**2)*(y**2),x**4, y**4,(x**3)*(y**3),x**5, y**5,(x**4)*(y**4),x**6, y**6,(x**5)*(y**5),x**7, y**7,(x**6)*(y**6),x**8, y**8,(x**7)*(y**7),x**9, y**9,(x**8)*(y**8),x**10, y**10,(x**9)*(y**9)), axis=-1)
    return X
