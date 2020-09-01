# Logistic Regression

*Aim:* The aim of this project was to implement all the machinery, including gradient descent, cost function, and logistic regression, of the various learning algorithms without module like sklearn, to get a deeper understanding of the fundamentals.

## Theoretical Overview

*Logistic Regression:* Logistic Regression belongs to major family of classification. The logistic function maps the entire real line and transforms it into space between 0 and 1. So, the function our model is finding converts all rational numbers the entire space of possibile values into something we can consider a probability. Contrary to it's name, logistic regression doesn't actually create a regression that is, it doesn't answer question with a real-valued number. It answers with a binary category, but it's called logistic regression because it's doing classification.

## For Manipulation
       import numpy as np
       import pandas as pd

## For Visualization
       import seaborn as sns
       import matplotlib.pyplot as plt 
       from pylab import rcParams
                  
## Name of the dataset
       DMV Written Tests
                   
## Shape of the dataset : 
       (100, 3)

## Tasks:
       Defining Logistic Sigmoid Function
       Computing Cost Function and Gradient
       Cost and Gradient at Initialization
       Gradient Descent
       Plotting the Convergence of Cost Function
       Plotting the decision boundary
       Predictions

## For Data Visualization, I used:
       scatterplot
       lineplot
