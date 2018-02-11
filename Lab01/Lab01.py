import numpy as np
import time
import math 
import matplotlib.pyplot as plt


#this is the logistic loss function which is defined in our labpdf
#takes in our model, features and our labels
def compute_loss(w,features,labels):
    #first were simply going to define a few variables
    #w_loss is our loss value for our model
    w_loss  = np.zeros(60000, float)
    #reassign features and labels to variables x and y 
    x = features
    y = labels
    
    #initialize total_sum which will add all loss values for our model (300 values because it has 300 dimensions)
    total_sum = 0.0
    
    #this is simply the lenght of our features (59,245)
    N = len(features)
    
    #now just go through the gradient function defined in the lab pdf
    #first negate all labels
    negative_y = y*-1
    
    #next do dot product of all features with our model
    dot_product = np.dot(x,w)
    
    #then mutiply that result with all our negated labels
    multiplication_operation = np.multiply(dot_product, negative_y)
    
    #now exponentiate that entire operation
    exponentiate = np.exp(multiplication_operation)
    
    #after that we simply have to take log of (1+exponentiate) and this will give us our loss for each dimension
    w_loss = np.log(1+exponentiate)
    
    #now we will take sum of all the dimensions and this will give us our overall loss
    total_sum = np.sum(w_loss)

    #and now we take the average by simply dividing by total number of features
    total_sum = total_sum/N
    
    #return that total value
    return total_sum

#this function performs the actual gradient descent algorithm
#takes in our model, features, labels, and the learning rate
#outputs our new model after applying the gradient
def compute_gradient(w, features, labels, learning_rate):
   #initialize gradient to a array of zeros of 300 dimensions 
    w_gradient = np.zeros(300, float)
    
    #initialize the lenght of our features 
    
    #reassign features and labels to variables x and y
    x = features
    y = labels
     
    #now just go through the gradient function defined in the lab pdf
    #first negate all labels
    negative_y = y*-1
    
    #next do dot product of all features with our model
    dot_product = np.dot(x,w)
    
    #then mutiply that result with all our negated labels
    multiplication_operation = np.multiply(dot_product, negative_y)
    
    #now exponentiate that entire operation
    exponentiate = np.exp(multiplication_operation)
    
    #now do fraction of our exponential/ (1 + exponentiation) as stated in the lab
    fraction = exponentiate/(1+exponentiate)
    
    #then multiply our entire fraction with our negatged labels
    parentheses = np.multiply(fraction,negative_y)
    
    #and our final gradient model is simply the dot product of everything in the parentheses and our features
    w_gradient = np.dot(parentheses,x)
    
    
    #then finally find new model, which is simply multiplying the gradient with our learning rate and subtracting
    #it from our original model. This is will slowly take us towards our convergence point/ local minimum
    new_w = w - (learning_rate*w_gradient)
    
    #return that new model 
    return new_w
 
def get_mini_batches(features,labels, batch_size):
    #initilize new lists to append the new mini_batches to
    x = []
    y = []
    
    #random_index = np.random.choice(len(x), len(y), replace = False)
    #x_shuffled = x[random_idxs, :]
    #y_shuffled = y[random_idxs]
    
    for i in range(0, len(labels), batch_size):
        x.append(features[i: i + batch_size])
        y.append(labels[i: i + batch_size])
    
    return x , y 
   
#this function is so that we can compute our gradient descent function on a specific number of iterations
# and then we can print out how much the the loss is at each iteration, and the total time since start of 
# our loop
def gradient_descent_runner(initial_w, features, labels, learning_rate, num_iterations, decay):
    #reinitialize our initial_w as w so we can have a changing model for each new iteration
    w = initial_w
    
    #reassign features and labels to variables x and y
    x = features
    y = labels
    
    #this is simply setting a list of losses to use for our plot
    loss_list = []
    
    #initilize our batch size to get our mini_batches
    batch_size = 50
    
    #call our mini_batch function that returns mini_batches x,y
    mini_batch = get_mini_batches(x,y,batch_size)
    
    #don't forget to set the first index to x(features) and second index to y(labels), 
    #because of the way the function is returning the mini_batches
    x_mini_batch = mini_batch[0] 
    y_mini_batch = mini_batch[1]
    
    #initilize j so that we can iterate though the mini_batches within our iterations
    #without going out of bound
    j = 0        
    
    #this is just to get the elapsed time 
    start = time.time()
    total_time = 0 
    
    #loop through all iterations
    for i in range(num_iterations):
        #this statment is when we reach 50 iterations, we will reset j 
        #otherwise we will have j = 51, which is out of bounds for the 
        #initialized batch_size
        if (i % len(x_mini_batch) == 0):
            j = 0
        
        #call compute gradient with our model,w features labels and learning_rate.
        #this function returns our new model,w 
        w = compute_gradient(w, x_mini_batch[j], y_mini_batch[j], learning_rate)
        learning_rate = learning_rate - decay
        
        #calculate loss of current mini_batch
        compute_losses = compute_loss(w,x_mini_batch[j],y_mini_batch[j])
        
        #increase j
        j += 1
        
        #appending the losses to a new list to use for our plot
        loss_list.append(compute_losses)
        
        #get our end time and calculate total_time to find elapsed time
        end = time.time()
        total_time = end-start
        
        #print out loss of new model at the current iteration
        print("Iteration = {0} , Loss: {1}, Elapsed Time: {2} ".format(i, compute_losses, total_time))
    
    #plot Loss versus Number of Iterations
    plt.plot(loss_list, color = "blue")
    plt.ylabel("Loss")
    plt.xlabel("Number of Iterations")
    plt.show()
    
    #once all iterations are complete, return the final model
    return w

def main():  
	
    #open file
    file = open("w8a.txt", "r")
    
    #create empty list for labels and features 
    labels = []
    features = []
	
    #go through every line in file
    for line in file:
        #split each line in file by white space
        mylist = line.split(" ")
        #append the first element, which is the +1/-1 labels to my labels list
        labels.append(int(mylist[0]))
        #create a list of 0's of dimension 300 for each example in my feature
        example = [0 for i in range(300)]
		
        #for all my examples, that goes from 1 to end of my list-1(-1 because we dont need first and last element),
        #last element is \n and first element is the labels, so our actual data is only 1 through (mylist-1)
        for i in range(1,len(mylist)-1):	
            #now each line has the form index:value, so we're going to split list by :
           indexAndValue = mylist[i].split(":")
           
           #now we will assign the 0th element of our new list ,which is the value before the colon,
           #which is our index to the variable index
           index = indexAndValue[0]
           #since our file/list is in str, we need to convert it to a int
           index = int(index)
           #then we will assign the value to that index in our examples list 
           example[index-1] = int(indexAndValue[1]) 
           
         #finally once we go through and do this with all examples, we will not append those examples to our features list
         #this is give us our total 59,245x300 list
        features.append(example)
    
    #next we will be using numpy to perform our matrix/vector/scalar operations, so we will not convert
    #our labels and features lists to arrays    
    labels = np.array(labels)
    features = np.array(features)
    
    #next we will now initialize our model, w to an array of 300 dimensions all with a float between 0.0 to 1.0
    initial_w = np.random.random_sample(300)
    
    #we do this step so we can have the values for our model between 0.0 to sqrt(1/150) instead of 0.0 to 1.0 
    initial_w = initial_w * (math.sqrt(1/150))
    
    #print(initial_w)
    #initialize our number of iterations and learning_rate and decay
    num_iterations = 1000
    learning_rate = 0.001
    decay = learning_rate/num_iterations
    
    #printing out initial loss 
    print("Starting Gradient Descent at loss = {}".format(compute_loss(initial_w, features,labels)))
    print("Running...")
    
    #now we will call this function which will run gradient descent which takes in
    # our initial model, features, labels, learning rage and the specified number of iterations
    w = gradient_descent_runner(initial_w, features, labels, learning_rate, num_iterations, decay)
    
    #once the entire gradient descent runner is complete, just print out total iterations and loss after all iterations
    #are complete
    print("After {0} iterations, loss = {1}".format(num_iterations, compute_loss(w, features, labels)))

    
main()
