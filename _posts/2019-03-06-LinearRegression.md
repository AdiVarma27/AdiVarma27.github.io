---
title: "Linear Regression (Gradient Descent) Python Implementation"
date: 2019-02-27
tags: [machine learning, linear regression, optimization, python, data science]
mathjax: "true"
---
In this post, we will use Gradient Descent to find the relationship between one independent/explanatory variable and the dependent/outcome variable.

### What is Gradient Descent ?
---
To simply put, gradient descent is an iterative first-order optimization algorithm to find minimum of a function. Like any other algorithm, gradient descent has its advantages and disadvantages. Let us express Linear Regression cost function in terms of a mathematical function, to understand how these two pieces tie together. 

The relationship between dependent and independent variable can be expressed as:   
**$$y = mx + c$$**,  where m corresponds to the slope of the line and c is the y-intercept.

We need to minimize the mean of sum of squared distance between predicted outcome of regression and the actual value of dependent variable (Mean Squared Error) in order to best-fit the regression line.

$$MSE = \displaystyle\frac{1}{n}\sum_{t=1}^{n}(h(x^i) - y^i)^2$$ 

The hypothesis $$h(x^i)$$ can be expressed as: $${\theta}_0 + {\theta}_1(x^i)$$, where $${\theta}_0$$ is the intercept and $${\theta}_1$$ is the slope of the best-fit line. We can express this as a cost-function whcih needs to be minimized, by updating the values of $${\theta}_0$$ and $${\theta}_1$$ simultaneously. Hence, the cost function is as follows:

$$J({\theta}_0, {\theta}_1) = \displaystyle\frac{1}{n}\sum_{t=1}^{n}(({\theta}_0 + {\theta}_1(x^i)) - y^i)^2$$

We need to find $${\theta}_0, {\theta}_1$$ such that the cost function $$J({\theta}_0, {\theta}_1)$$ is minimized (Least Mean Squared Error)

### Gradient Descent Algorithm
---
<img src="{{site.url}}{{site.baseurl}}/images/post1/img1.jpeg">
where $${\alpha}$$ is the 'learning rate', which is a scalar quantity that defines the step size, and is multiplied with the slope/ partial derivative of the cost function curve. 

<img src="{{site.url}}{{site.baseurl}}/images/post1/img2.jpeg">

### Python Implementation
---
Below is the implementation of gradient descent, with default values for $${\theta}_0$$, $${\theta}_1$$ set to 0. Number of iterations and alpha set to 1000 and 0.01 respectively.
```python
def GradientDescent(X, y, theta0=0, theta1=0, max_iterations=1000, alpha=0.01):
    
    # iteration upto max_iterations
    for num_iterations in range(max_iterations):
        
        # predicted outcome vector y_pred
        y_pred = theta0 + theta1*X
        
        # computed cost
        total_cost = (1/len(y))*sum([val**2 for val in (y-y_pred)])
        
        # partial derivatives 
        pdev_theta0 = (2/len(y))*sum(y_pred - y)
        pdev_theta1 = (2/len(y))*sum(X*(y_pred - y))

        # update parameters 
        theta0 -= alpha*pdev_theta0
        theta1 -= alpha*pdev_theta1
        
        print("slope {}, intercept {}, cost {}".format(theta1, theta0, total_cost))
        
        # plotting every updated regression line
        plt.plot(X, theta0 + theta1*X, c='g')
        
    # plotting actual X, y points, along with final best-fit line in red
    plt.scatter(X, y, c='r') 
    plt.plot(X, theta0 + theta1*X, c='r', lw=3)
    
    return theta0, theta1, theta0 + theta1*X

X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([5,6,9,11,13,11,11,13,22,29])
# function call
theta0, theta1, y_pred = GradientDescent(X, y)
```

Below, we observe that the red line is the final Regression line (best fit) using Gradient descent, with intercept theta0 = 1.449 and slope theta1 = 2.099.

<img src="{{site.url}}{{site.baseurl}}/images/post1/graph.jpeg">

Tuning the learning rate $${\alpha}$$ is an important part of Gradient Descent, or the cost function $$J({\theta}_0, {\theta}_1)$$ may or maynot converge everytime. However, the learning rate need not be adjusted after every iteration, as the rate of change/ slope would be initially high, and eventually starts decreasing over iterations. 

As we observe below, in Figure 1, the $${\alpha}$$ is small, the cost function is reduced iteratively and finally reaches local minima. 

For a higher $${\alpha}$$, the cost function does not converge, and overshoots the value. Hence, learning rate parameter needs to be tuned properly.

<img src="{{site.url}}{{site.baseurl}}/images/post1/img3.jpeg">

For extremely large datasets, Gradient Descent makes sense as it is computationally less expensive when compared to Normal Equation methodology. In the next post, we will take a look at using other Optimization Techniques such as L1 (Lasso) and L2 (Ridge) Regularization, and various other Gradient descent techniques.
