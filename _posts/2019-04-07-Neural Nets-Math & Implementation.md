
### This post is to help provide an intuitive understanding of a (dumb) 2-Layered Neural Network.

* In this following post, we will start off with building a 1-Layer Neural Net, and how it maps to Logistic Regression. We will test our functions/ model with multiple datasets. 


* Later, we will improve on our initial model and include a single hidden layer, and see how our performance increases in classification task. We will also tune/ tweak the number of hidden nodes in our hidden layer, and try to gain an intuition about how Neural Nets could be used efficiently over traditional classification models.


* In the next post, once we understand the basic building blocks of non-linear functions, along with a generalized forward and backward propagation steps, we will construct our very own Deep Neural Network package, with Model() and Layer().


* My 2 cents: Don't worry about optimization yet; details about activation functions, mini-batch gradient descent, RMSprop, Adams Optimization and other techniques are used to fine-tune model performance, which will be covered in a different post.


```python
#importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

### Activation Functions

Let's load the activation functions we will use in this classification task. We use Sigmoid function to map our input to [0,1] in binary classification task. 

* The final output layer uses the Sigmoid function (remember that the activation function of Logistic regression is based on sigmoid fucntion).


* Tanh activation function is similary to Sigmoid function, but y-values go below 0. (i.e., [-1,1]), derivates are steeper when compared to Sigmoid function.


* Let's take an array from -3 to +3 with step size of 0.05. After 'activating' the input with sigmoid, tanh and relu, we see that tanh is a transformation of the sigmoid activation, whereas relu gives maximum of 0 and the value itself.


```python
# Sigmoid function -> maps input to [0,1]
def sigmoid(X):
    return (1/(1+np.exp(-X)))

# Tanh fucntions -> maps input to [-1,1]
def tanh(X):
    return np.tanh(X)

# Rectifier which maps to [0,max(x)]
def relu(X):
    return np.maximum(0,X)
```


```python
# input vector to activation functions
x = np.arange(-3,3,0.05)

# activation steps
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 2 plots 
plt.figure(figsize=(14,6))
plt.subplot(121), plt.scatter(x, y_sigmoid, label='sigmoid'), plt.scatter(x, y_tanh, label='tanh')
plt.grid(), plt.legend(loc='upper left'), plt.title("Sigmoid vs Tanh")
plt.xlabel('x'), plt.ylabel('activated x')

plt.subplot(122)
plt.scatter(x, y_sigmoid, label='sigmoid'), plt.scatter(x, y_tanh, label='tanh')
plt.scatter(x, y_relu, label='relu'), plt.grid()
plt.legend(loc='upper left'), plt.title("Sigmoid vs Tanh vs Relu"), plt.xlabel('x'), plt.ylabel('activated x')

plt.show()
```


![png](/images/post3/output_5_0.png)


* From the above graphs, we understand why Sigmoid is used for our final Layer (or Logistic Regression) in case of Binary classification; to predict either 0 or 1. Now, lets load our dataset and explore it further.


## 1-Layer Neural Net (Logistic regression)

Let's load a simple dataset with two classes, and see how a basic 1-Layer Neural Network, is the same as a Logistic Regression Task, and how activation functions are used to activate linear combinations of input vectors and parameters. 


```python
# take features into X and targets into y
from sklearn.datasets import make_moons

# Dataset, set sample size
X, y = make_moons(n_samples=800, noise=0.5)

# Splitting into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.5, random_state=27)

# working with transposes for ease of matrix manipulation
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# shapes
X.shape, y.shape
```




    ((800, 2), (800,))




```python
# Viz of input data
plt.scatter(X[:,0], X[:, 1], c=y)
```




    <matplotlib.collections.PathCollection at 0x1a17703470>




![png](/images/post3/output_9_1.png)


* Now, let us consider the input feature matrix X-transpose, with shape (2, 800), represented as the zeroth layer/ input layer of our Neural Network. As we have two features (x1, x2) for our input data, we would have 2-weights (w1, w2), along with the term b, as the parameters of our model.

### Propagation Step:

* The function initialize_weights returns (w, b), where w is the weight vector of shape (2,1) and b is zero. It uses numpy random.rand function to generate values with seed set to 27.


* We form a linear combination of (x1, x2), by using randomly initialized weights (w1, w2, b), to form Z.


* We then activate Z, by using Sigmoid function, which returns a probability of belonging to class 1, from [0,1].


* Then, we calculate the cost/ loss with the parameters (w1, w2, b). However, we need to update our parameters such that the Log loss is minimized, read <a href="https://adivarma27.github.io/LogisticRegressionCost/#">this article</a> for detailed explanation. Hence, we need to update the parameter vectors w and b, in such a way that the Log loss is minimized. We need to perform gradient descent, and find the slope of Loss function, with respect to each of the parameters.


* Below is a flow diagram of the 1-Layer NN.


Note: In the following image below, w1 and w2 are vectorized into w for matrix multiplication.


<img src="/images/post3/img1.jpeg" width="700">


### Back-Propagation Step:

* As this is a Single Layer Neural Net, there is just one back-propagation step involved. Each of the input parameter vector (w and b, in this case), decreases its value by a learning rate '$\alpha$' times the derivative of loss function with respect to itself. Hence, the learning_rate in itself is a hyperparameter and needs to be initialized properly, for efficient training process.


* Below is the math behind finding derivatives of cost function with respect to parameters, which will aid with update parameter step.


<img src="/images/post3/img2.jpeg" width="800">

<img src="/images/post3/img3.jpeg" width="800">

* In the back-propagation step, we need to update only our weight parameters (w, b); however to calculate derivative of cost function with respect to parameter, we need to propagate through the Linear combination (Z), as well the activated output (A).


* As this Neural-Net has a single layer, the back-propagation is straight forward. As the number of hidden layers increase, we can generalize the forward & backward propagation step.


* Once the output is computed, the predict function takes in the y vector which contains probabilities of belonging to class 0 or 1 and assigns class based on 0.5 threshold.


```python
# function to initialize weights
def initialize_weights(X, seed=27):
    np.random.seed(27)
    w = np.random.rand(X.shape[0],1)
    b = 0
    
    return w, b

# function to predict class on 0.5 threshold
def predict(y):
    y_list = []
    y = y[0]
    for i in range(0,len(y)):
        if y[i] > 0.5:
            y_list.append(1)
        else:
            y_list.append(0)
        
    return y_list

# See post https://adivarma27.github.io/LogisticRegressionCost/# for detailed explanation
def train(X, y, w, b, max_iterations=1000, learning_rate=0.01):

    m = X.shape[1]
    learning_rate = learning_rate
    max_iterations = max_iterations
    cost_list, train_acc_list, test_acc_list, w1_list, w2_list, b_list = [], [], [], [], [], []

    # iteration
    for iteration in range(max_iterations):

        # linear-combination, activation step
        Z = np.dot(w.T, X) + b
        A = sigmoid(Z)      

        # compute cost 
        cost = -(1/m)*np.sum(y*np.log(A) + (1-y)*np.log(1-A))
        cost_list.append(cost)

        # back-prop
        # derivative with respect to cost function
        dw = (1/m)*np.dot(X, (A-y).T)
        db = (1/m)*np.sum(A-y)

        # parameter update step
        w = w - learning_rate*dw
        b = b - learning_rate*db

        w1_list.append(w[0])
        w2_list.append(w[1])
        b_list.append(b)
        
        train_acc_list.append(accuracy_score(predict(A), y.squeeze()))

        test_acc_list.append(PredictClass(X_test, w, b))
        
    return cost_list, train_acc_list, test_acc_list, w1_list, w2_list, b_list

# Forward prop class prediction
def PredictClass(X, w, b):
    y_pred_list = []
    A = sigmoid(np.dot(w.T, X) + b)
    ypred = predict(A)
    y_pred_list.append(ypred)

    return y_pred_list

# Function to rpedict test data
def PredTestData(test_acc_list):
    y_pred_list = []
    for i in range(0,len(test_acc_list)):
        y_pred_list.append(accuracy_score(test_acc_list[i][0], y_test))
        
    return y_pred_list
```

### For each iteration:

Propagation Steps:

1. compute linear combination of weights and input vector
2. activate input vector and store in vector A
3. compute cost with the current weights

Back-Prop Steps:

4. compute derivates/ slope of cost function with respect to corresponding weights
5. Update weights

Prediction:

6. Predict class


```python
# hyperparameters alpha, max_iterations
learning_rate = 0.05
max_iterations = 1000

# Initialize weights
w, b = initialize_weights(X_train, seed=27)

# train over X_train, y_train
cost_list, train_acc_list, test_acc_list, w1_list, w2_list, b_list = train(X_train, y_train, w, b)

# testing data prediction task
test_pred = PredTestData(test_acc_list)
```


```python
# subplots
plt.figure(figsize=(14,6)), plt.subplot(121)
plt.scatter(np.arange(0, max_iterations), cost_list), plt.xlabel('# of Iterations'), plt.ylabel('Cost')
plt.title('Scatter plot of Decreasing Cost over number of iterations'), plt.grid()

plt.subplot(122), plt.scatter(np.arange(0,max_iterations), train_acc_list, label='Training Accuracy')
plt.scatter(np.arange(0,max_iterations), test_pred, label='Testing Accuracy')
plt.legend()
plt.xlabel('# of Iterations'), plt.ylabel('Accuracy')
plt.title('Scatter plot of Training Accuracy over number of iterations'), plt.grid()
plt.show()
```


![png](/images/post3/output_14_0.png)


#### As we see from the above graphs, as the number of iterations reaches 1000, we observe saturating Training & Testing Accuracy, and we achieve a testing accuracy of ~ 77 %. In the above 1-Layer Neural Net (Logistic Regression), we do a decent job, where the input features are just linear combinations along with weights w1, w2, b. 

#### Below, we plot the weights w1, w2 and see how they converge by our gradient descent step, with x-axis as w1, w2 values and y-axis as cost function. Initially, the gradient step is higher; in later iterations the step size decreases.


```python
plt.scatter(w1_list, cost_list, label='w1'), plt.scatter(w2_list, cost_list, label='w2'), 
plt.legend(), plt.xlabel('w1, w2 values'), plt.ylabel('Cost Function')
```




    (<matplotlib.legend.Legend at 0x1a17296d68>,
     Text(0.5, 0, 'w1, w2 values'),
     Text(0, 0.5, 'Cost Function'))




![png](/images/post3/output_16_1.png)


## 2-Layer Neural Net (1-Hidden Layer, n-Hidden Units)

* In the above exercise, we saw how Logistic Regression, is a **simple Feed-forward neural network**. Now, lets include a hidden layer in our Neural Net, and select number of hidden units as a hyperparameter in our model.


* We can also choose between different activation functions (relu or tanh) functions for activating our hidden layer, after which we will activate the final layer with Sigmoid function, to obtain probability of belonging to class 1. This is where we understand the advantages of Neural Nets over Logistic Regression model.


* Let us choose a complex dataset to understand the power of hidden layers/ hidden units in our Neural net model.


```python
# Complex dataset

# source: https://github.com/rvarun7777/Deep_Learning/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/planar_utils.py
# function to load our dataset
def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y
```


```python
# take features into X and targets into y
X, y = load_planar_dataset()

# shapes
X.shape, y.shape
```




    ((2, 400), (1, 400))



* We obtained around 75-80 % testing accuracy on previous dataset by using 1 Layer NN. We observe that the same model has only 50 % accuracy on the new dataset. (Model has just w1, w2 & b to capture complex decision boundary)


```python
plt.scatter(X[0,:], X[1,:], c=y.ravel())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5)

clf = LogisticRegressionCV()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred)*100, '% accuracy')
```

    /anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)


    54.0 % accuracy



![png](/images/post3/output_22_2.png)


## Why is 2-Layer NN better than 1-Layer NN/ Any traditional Machine Learning Model ?

* From the above, we see that traditional single layer NN/ Logistic Regression gives only around 50 % accuracy. Clearly, there are hidden patterns which the model is unable to capture. The model needs more number of parameters to be able to capture the pattern in our dataset, to form/ contruct non-linear complex boundaries.


* The **power** of hidden layers lies in the fact that these complex decision boundaries can be captured; and generally speaking, more the number of hidden units in a layer, better we can train the data. Previously, we had only w1, w2 & b to tune to find the best decision boundary. As the number of layers and number of nodes increase, we have many more parameters to optimize. The weights which are randomly initialized, start 'self-adjusting' or 'self-correcting', where the combinations of different weights in different nodes and layers, try to minimize log-loss. After multiple iterations, the layers start recognizing patterns in the data and start self-correcting. Finally, we are left with many more number of parameters/ weight vectors, which could overfit to the training data. Essentially, each parameter (on each layer), tries to find the derivative of cost/ loss function with respect to itself, to update the parameter value. 


* Since we have one hidden layer and 4 hidden units, our parameters are now vectors W1, W2, B1, B2, where W1, B1 correspond to the input layer (input layer to hidden layer), and W2, B2 (weights corresponding to hidden layer to output). W1 = [w1, w2] (input layer), W2 = [w1, w2, w3, w4] (4 hidden nodes).


<img src="/images/post3/img4.jpeg" width="800">


* Below are the functions **initialize_weights_2layers(X, n_hidden_units)** and **trainHidden(params, X, max_iterations)** to initialize weights vector based on number of hidden features in second layer.


```python
# function to initialize parameters 
def initialize_weights_2layers(X, n_hidden_units = 10, seed=27):
    np.random.seed(seed=seed)
    params = {}
    
    W1 = np.random.randn(n_hidden_units, X.shape[0])*0.01
    B1 = np.zeros(shape=(n_hidden_units, 1))
    W2 = np.random.randn(1,n_hidden_units)*0.01
    B2 = np.zeros(shape=(1,1))
    
    params['W1'], params['W2'], params['B1'], params['B2'] = W1, W2, B1, B2
    
    return params

params = initialize_weights_2layers(X, n_hidden_units=2)
W1, W2, B1, B2 = params['W1'], params['W2'], params['B1'], params['B2']
params
```




    {'W1': array([[ 0.01285605, -0.00303553],
            [ 0.00619076,  0.00395999]]),
     'W2': array([[ 0.00223406, -0.00054339]]),
     'B1': array([[0.],
            [0.]]),
     'B2': array([[0.]])}




```python
# function to train 2 Layered NN
def trainHidden(params, X, max_iterations=8000):
    W1, W2, B1, B2 = params['W1'], params['W2'], params['B1'], params['B2']
    m = X.shape[1]
    learning_rate = 1
    acc = []
    
    # iteratiing
    for iteration in range(max_iterations):
        
        # linear combination and first activation
        Z1 = np.dot(W1, X) + B1
        A1 = tanh(Z1)
        
        # linear combination and second activation
        Z2 = np.dot(W2, A1) + B2
        A2 = sigmoid(Z2)
        
        acc.append(predict(A2))
        
        # calculating cost
        cost = -(1/m)*np.sum(np.multiply(np.log(A2), y) + np.multiply((1 - y), np.log(1 - A2)))

        # backprop step
        dz2 = A2 - y
        dw2 = (1/m)*np.dot(dz2,A1.T)
        db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
        dz1 = np.multiply(np.dot(W2.T,dz2),1-np.power(A1, 2))
        dw1 = (1/m)*np.dot(dz1,X.T)
        db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)

        # parameter update step
        W1 = W1 - learning_rate*dw1
        B1 = B1 - learning_rate*db1
        W2 = W2 - learning_rate*dw2
        B2 = B2 - learning_rate*db2
    
    return acc
```


```python
# function to train over 'n' hidden units
def trainhiddenUnits(n_hidden_units):

    params = initialize_weights_2layers(X, n_hidden_units)
    acc = trainHidden(params, X=X, max_iterations=5000)

    accuracy = []

    for i in range(0,len(acc)):
        accuracy.append(accuracy_score(acc[i], y.squeeze()))

    return accuracy
```

Below are the accuracies after including one hidden layer (with various number of hidden units). We observe that by using 3-hidden units, the model is able to capture the complexity and able to predict pretty well. (~ 40% increase in accuracy when compared to 1 Layered Neural Net).


```python
# list of accuracies for n hidden units
accuracies_ = []

for n_hidden_units in range(1,5):
    accuracies_.append(trainhiddenUnits(n_hidden_units))

# Plots for various number of hidden units
plt.figure(figsize=(16,8))
plt.scatter(np.arange(0,5000), accuracies_[0], label='1 hidden unit')
plt.scatter(np.arange(0,5000), accuracies_[1], label='2 hidden units')
plt.scatter(np.arange(0,5000), accuracies_[2], label='3 hidden units')
plt.scatter(np.arange(0,5000), accuracies_[3], label='4 hidden units')
plt.xlabel('# of iterations')
plt.ylabel('Accuracy')

plt.legend()
plt.grid()
```


![png](/images/post3/output_28_0.png)


### Why does predictive power saturate at 3 Hidden Units ? Why not 100 units if it performs better ? 

* As you see below, once we reach the decision boundaries which can form separation between the two classes, predictive power saturates, beyond which it overfits to the data.


* For the same data, 1 Layered NN performed poorly (~45 % testing accuracy), as it was not able to understand this pattern, which looks simple to the human eye.


```python
plt.figure(figsize=(12,8))
plt.scatter(X[0,:], X[1,:], c=y.ravel())
plt.plot([-4.5, 4.5], [-0.8, 0.8], 'k-', lw=2)
plt.plot([-4.5, 4.5], [-3.1, 3.1], 'k-', lw=2)
plt.plot([-4.5, 4.5], [3.25, -3.2], 'k-', lw=2)
```




    [<matplotlib.lines.Line2D at 0x1a18700898>]




![png](/images/post3/output_30_1.png)


* Now, lets try another dataset, where decision boundary can be visually understood. First, we use Logistic Regression to see if the model can understand the obvious pattern of different classes.


```python
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(n_samples=400,factor=0.5,noise = 0.1)

plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap = 'winter')
plt.title('Nonlinear Data')

X, y = X.T, y.T
```


![png](output_32_0.png)


* We observe that the accuracy is only around 50 %. Hence, just W and b vectors are not enough to model the non-linearity that exists in the data.


```python
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5)

clf = LogisticRegressionCV()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred)*100, '% accuracy')
```

    60.0 % accuracy


    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)


### Hence, we traing the data using 2-Layered Neural Net, over various number of hidden layers, and find that 3 hidden units can predict with 100 % accuracy.


```python
# list of accuracies for n hidden units
accuracies_ = []

for n_hidden_units in range(1,4):
    accuracies_.append(trainhiddenUnits(n_hidden_units))

# Plots for various number of hidden units
plt.figure(figsize=(16,8))
plt.scatter(np.arange(0,5000), accuracies_[0], label='1 hidden unit')
plt.scatter(np.arange(0,5000), accuracies_[1], label='2 hidden units')
plt.scatter(np.arange(0,5000), accuracies_[2], label='3 hidden units')

plt.xlabel('# of iterations')
plt.ylabel('Accuracy')

plt.legend()
plt.grid()
```


![png](/images/post3/output_36_0.png)


#### Deep Layered Neural Nets can now out perform better than human beings at classification tasks. We are yet to tune all the hyper-parameters, use optimization techniques to help parameters converge faster, and also look at Convoluted/ Recurrent layers to improve performance. 
