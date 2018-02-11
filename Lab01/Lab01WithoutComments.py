# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:16:51 2018

@author: Twinkle
"""

import numpy as np
import time
import math 
import matplotlib.pyplot as plt
import mkl 

def compute_loss(w,features,labels):
    w_loss  = np.zeros(60000, float)
    x = features
    y = labels

    total_sum = 0.0

    N = len(features)

    negative_y = y*-1
    dot_product = np.dot(x,w)
    multiplication_operation = np.multiply(dot_product, negative_y)
    exponentiate = np.exp(multiplication_operation)
    w_loss = np.log(1+exponentiate)
    total_sum = np.sum(w_loss)
    total_sum = total_sum/N

    return total_sum

def compute_gradient(w, features, labels, learning_rate):

    w_gradient = np.zeros(300, float)
    x = features
    y = labels

    negative_y = y*-1
    dot_product = np.dot(x,w)
    multiplication_operation = np.multiply(dot_product, negative_y)
    exponentiate = np.exp(multiplication_operation)
    fraction = exponentiate/(1+exponentiate)
    parentheses = np.multiply(fraction,negative_y)
    w_gradient = np.dot(parentheses,x)
    new_w = w - (learning_rate*w_gradient)

    return new_w
 
def get_mini_batches(features,labels, batch_size):
    x = []
    y = []
        
    for i in range(0, len(labels), batch_size):
        x.append(features[i: i + batch_size])
        y.append(labels[i: i + batch_size])
    
    return x , y 

def gradient_descent_runner(initial_w, features, labels, learning_rate, num_iterations, decay):
    w = initial_w
    
    x = features
    y = labels
    loss_list = []
    batch_size = 50
    
    mini_batch = get_mini_batches(x,y,batch_size)
    x_mini_batch = mini_batch[0] 
    y_mini_batch = mini_batch[1]
    
    j = 0
    
    start = time.time()
    global total_time 
    total_time = 0

    for i in range(num_iterations):
        if (i % len(x_mini_batch) == 0):
            j = 0

        w = compute_gradient(w, x_mini_batch[j], y_mini_batch[j], learning_rate)
        learning_rate = learning_rate - decay

        c_l = compute_loss(w,x_mini_batch[j],y_mini_batch[j])
        j += 1
        
        loss_list.append(c_l)
        
        end = time.time()
        total_time = end - start
        
        print("Iteration = {0} , Loss: {1}, Elapsed Time: {2} ".format(i, c_l, total_time))
    

    plt.plot(loss_list, color = "blue")
    plt.ylabel("Loss")
    plt.xlabel("Number of Iterations")
    plt.show()
    
    return w

def main():  
	
    file = open("w8a.txt", "r")
    
    labels = []
    features = []
	
    for line in file:
        mylist = line.split(" ")
        labels.append(int(mylist[0]))
        example = [0 for i in range(300)]
		
        for i in range(1,len(mylist)-1):	
           indexAndValue = mylist[i].split(":")
           
           index = indexAndValue[0]
           index = int(index)
           example[index-1] = int(indexAndValue[1]) 
           
        features.append(example)
    
    labels = np.array(labels)
    features = np.array(features)
    
    initial_w = np.random.random_sample(300)
    initial_w = initial_w * (math.sqrt(1/150))
    
    num_iterations = 1000
    learning_rate = 0.001
    decay = learning_rate/num_iterations
    mkl.set_num_threads(4)
    
    print("Starting Gradient Descent at loss = {}".format(compute_loss(initial_w, features,labels)))
    print("Running...")
 
    w = gradient_descent_runner(initial_w, features, labels, learning_rate, num_iterations, decay)
    
    print("After {0} iterations, loss = {1}, Elapsed Time = {2} ".format(num_iterations, compute_loss(w, features, labels), total_time))
    
main()
