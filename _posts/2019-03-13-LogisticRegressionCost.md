---
title: "Logistic Regression, Log Loss, Cost Function Derivation, Python Implementation"
date: 2019-02-27
tags: [machine learning, logistic regression, optimization, python, data science, classification, log loss]
mathjax: "true"
---
In this post, we will look at log loss, derivation of logistic cost function, to implement a basic calssification task.

### What is Logistic Regression ?
---
A Binary Logistic Regression model classifies input vectors into classes, by calculating probability of belonging to class 1. Given the input vector $$X$$ that belongs to any real value, the output label $$y$$, is either 0 or 1. 

The Sigmoid/ Logit function is used to to generate a probability score $$[0,1]$$, of belonging to class 1, by passing the  the linear combination of weights and vectors, i.e., $${\theta}_0$$ + $${\theta}_1X$$, through the Sigmoid function. As we need hypothesis $$h(x)$$, where $$0<=h(x)<=1$$ , our new hypothesis becomes 

$$h(x) = g({\theta}^TX)$$ = $$P(y=1/x;{\theta})$$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

where $$g$$ maps the input to $$[0,1]$$. Below is the Python code to map input vector X, along with input parameters $${\theta}_0$$, $${\theta}_1$$ to a probability.

```python
def Logit(X, theta_0, theta_1):
    return 1/(1 + np.exp(-(theta_0 + theta_1*X)))
```
### Log Loss Intuition
---
Log Loss can be calulated by penalizing heavily when the predicted probability is away from the actual label. If actual label $$y = 1$$, and if the predicted output is close to $$1$$, the cost of the loss function is close to $$0$$; If the predicted probability is closer to $$0$$, the cost must increase/ penalized heavily. 

$$J(h(x),y)$$ =  $$-ylog(h(x))-(1-y)log(1-h(x))$$

If $$y=1$$, $$J(h(x),y) = -log(h(x)$$

If $$y=0$$, $$J(h(x),y) = -log(1-h(x))$$

### Cost Function Derivation
---
We use gradient descent to find the optimal parameters $${\theta}_0$$, $${\theta}_1$$ to reduce the Cost function $$J({\theta})$$. Below is the gradient descent algorithm:
 
<img src="{{site.url}}{{site.baseurl}}/images/post2/img2.jpeg">
where α is the ‘learning rate’, which is a scalar quantity that defines the step size, and is multiplied with the slope/ partial derivative of the cost function curve. We need to compute partial derivatives of the cost function with respect to parameters for the update step.
 
<img src="{{site.url}}{{site.baseurl}}/images/post2/img1.jpeg">

Hence, the partial derivative of the cost function in Logistic Regression looks similar to the one of Linear Regression. 
 


### Python Implementation
---
 Below is the implementation of gradient descent, with default values for $${\theta}_0$$, $${\theta}_1$$ set to 0. Number of iterations and alpha set to 1000 and 0.1 respectively.
 ```python 
 import numpy as np

# sigmoid function/ outputs probability 
def Logit(X, theta_0, theta_1):
    return 1/(1 + np.exp(-(theta_0 + theta_1*X)))

# Logistic Rgression function
def LogisticRegression(X, y, theta_0=0, theta_1=0, max_iterations=1000, alpha=0.01):

    # current iteration until max_iterations
    for iteration in range(max_iterations):
        
        # partial derivatives
        p_dev_theta0 = (1/len(X))*np.sum(Logit(X, theta_0, theta_1) - y)
        p_dev_theta1 = (1/len(X))*np.sum((Logit(X, theta_0, theta_1) - y)*X)
        
        # parameter update step
        theta_0 = theta_0 - alpha*p_dev_theta0
        theta_1 = theta_1 - alpha*p_dev_theta1
        
        # total cost J
        total_cost = -(1/len(X))*(np.sum(y*np.log(Logit(X, theta_0, theta_1)) + 
                                         (1-y)*(np.log(1-Logit(X, theta_0, theta_1)))))
        
        print(theta_0, theta_1, total_cost)
        
    return theta_0, theta_1

# sample dataset
X = np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50])
y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

# function call
theta_0, theta_1 = LogisticRegression(X, y, max_iterations=1000, alpha=0.1)
```


From the graph below, we see that the parameters $${\theta}_0$$, $${\theta}_1$$ are tuned to reduce the log loss between predicted probabilit and actual class label. 
 <img src="{{site.url}}{{site.baseurl}}/images/post2/img3.jpeg">

The points in yellow belong to class 0 and points in blue belong to class 1. We see that the decision boundary separates the two classes and the final decision boundary (in red) minimizes the cost function, and find optimal parameters $${\theta}_0$$, $${\theta}_1$$.