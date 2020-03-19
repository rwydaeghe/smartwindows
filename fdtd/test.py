# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:46:46 2020

@author: rbnwy
"""
import numpy as np
import time
class Thing:
    def __init__(self,e):
        self.otherthing = e
    def __add__(self,y):
        return self.otherthing+y.otherthing
    def add(self,y):
        return self.otherthing+y.otherthing

things=Thing(np.array([1,2,3]))
print(things.otherthing)
a=np.arange(1,10000000)
t=time.time()
Thing.add(thing1,thing2)
print(time.time()-t)
t1=time.time()-t
a+a
print(time.time()-t)