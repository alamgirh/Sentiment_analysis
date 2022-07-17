# Sentiment Analysis using Logistic Regression
We will implement logistic regression for sentiment analysis on tweets. Given a tweet, we will decide if it has a positive sentiment or a negative one. Steps of this project are: 

* Extract features for logistic regression from given some text/tweets
* Implement logistic regression from scratch
* Apply logistic regression on a natural language processing task
* Test using logistic regression
* Perform error analysis


## Logistic regression: regression and a sigmoid

Logistic regression takes a regular linear regression, and applies a sigmoid to the output of the linear regression.

Regression:
$$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N$$
Note that the $\theta$ values are "weights". If you took the deep learning specialization, we referred to the weights with the 'w' vector.  In this course, we're using a different variable $\theta$ to refer to the weights.

Logistic regression
$$ h(z) = \frac{1}{1+\exp^{-z}}$$
$$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N$$
We will refer to 'z' as the 'logits'.
