import numpy as np

def compute_loss(w,x,y):
    w_loss = 0 

   
    negative_y = y*-1
    dot_product = np.dot(x,w)
    multiplication_operation = np.multiply(dot_product, negative_y)
    exponentiate = np.exp(multiplication_operation)
    w_loss += np.log(1+exponentiate)
    #print(w_loss)
        
def compute_gradient(w, x, y, learning_rate):
   
    w_gradient = np.zeros(2, float)
    N = len(x)
    
    
    negative_y = y*-1
    dot_product = np.dot(x,w)
    multiplication_operation = np.multiply(dot_product, negative_y)
    exponentiate = np.exp(multiplication_operation)
    fraction = exponentiate/(1+exponentiate)
    parentheses = np.multiply(fraction,negative_y)
    w_gradient += (1/N)*np.dot(parentheses,x)
    print(w_gradient)
    #for j in range(len(x)):
    #       w_gradient[j] += (1/N)*np.dot(x[j],parentheses)
    #w_gradient[w] = np.dot(parentheses,x)
    #w_gradient[w] += (1/N)*np.dot(x[:,x],parentheses)
    #new_w = w - (learning_rate*w_gradient)
    

    

x = np.array([[1,2], [1,2], [1,2]])
y = np.array([1,2,-1])
w = np.array([1, 1])

learning_rate = 0.0001
compute_loss(w,x,y)
compute_gradient(w,x,y,learning_rate)

#print(compute_loss)


