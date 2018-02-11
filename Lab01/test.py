# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:12:19 2018

@author: Twinkle
"""

import numpy as np 
import time
import mkl 

a = np.random.random_integers(50)
b = np.random.random_integers(50)

start = time.time()
total_time = 0 
total_time_with_threads = 0

for i in range(5000000):
    dot_product = np.dot(a,b)
end = time.time()
total_time = end-start

start = time.time()
mkl.set_num_threads(2)    
for i in range(5000000):
    dot_product = np.dot(a,b)
end = time.time()
total_time_with_threads = end-start
    
print(dot_product)
print(total_time)
print(total_time_with_threads)
print(total_time - total_time_with_threads)