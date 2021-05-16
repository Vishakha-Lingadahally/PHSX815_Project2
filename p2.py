#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:48:22 2021

@author: vishakha
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:29:09 2021

@author: vishakha
"""

import numpy as np
import math

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')

# Importing pandas
import pandas as pd
from scipy.stats.distributions import chi2
# Begin by flipping coins

flip_1 = np.random.rand()
flip_2 = np.random.rand()
flip_3 = np.random.rand()
print('The resultant probabilities are %s, %s and %s' %(flip_1, flip_2, flip_3))

print('The results of the flips are:')
flips = [flip_1, flip_2, flip_3]
for flip in flips:
    if flip < 0.5:
        print("Heads")
    else: 
        print("Tails")
        
        
        
# Test if our coin flipping algorithm is fair.
n_flips = 1000
p = 0.5  # Our expected probability of a heads.

# Flip the coin n_flips times.
flips = np.random.rand(n_flips)

# Compute the number of heads.
heads_or_tails = flips < p  # Will result in a True (1.0) if heads.
n_heads = np.sum(heads_or_tails)  # Gives the total number of heads.

# Compute the probability of a heads in our simulation.
p_sim = n_heads / n_flips
print('Predicted probability = %s. Simulated probability = %s.' %(p, p_sim))

# Define our step probability and number of steps.

# Hypothesis 1:

step_prob = 0.5  # Can step left or right equally.
n_steps = 100 # Essentially number of steps.

    
# Set up a vector to store our positions.

position = np.zeros(n_steps)

# Loop through each time step.
for i in range(1, n_steps):
    # Flip a coin.
    flip = np.random.rand()
    
    # Figure out which way we should step.
    if flip < step_prob:
        step = -1 # To the 'left'.
    else:
        step = 1# to the 'right'.
        
    # Update our position based on where we were in the last time point. 
    position[i] = position[i-1] + step
    
#print(position[i])

# Number of steps taken to the left in our first hypothesis
nl=int((n_steps-position[i])/2)
print("The number of steps taken to the left in our first hypothesis is %s.Therefore, the number of steps taken to the right is %s." %(nl, (n_steps-nl)))
# Make a vector of time points.
steps = np.arange(0, n_steps, 1)  # Arange from 0 to n_steps taking intervals of 1.
 
d={'steps':np.array(steps), 'position':np.array(position)}
df=pd.DataFrame(d)

# Likelihood for binomial distribution

P=np.log(((math.factorial(n_steps))*((step_prob)**(n_steps-nl))*((1-step_prob)**(nl)))/((math.factorial(nl))*(math.factorial(n_steps-nl))))
#P=np.log(((math.factorial(n_steps))*((step_prob)**(100-nl))*((1-step_prob)**(nl)))/(((math.factorial(nl)))*(math.factorial(100-nl))))
print("The log likelihood for our first hyposthesis is %s" %(P))
#print(steps)
#print(position)
# Plot it!
plt.plot(steps, position)
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()



# Make a vector of time points.



# Perform the random walk multiple times. 
n_simulations = 1000

# Make a new position vector. This will include all simulations.
position = np.zeros((n_simulations, n_steps))


# Redefine our step probabilities just to be clear. 
step_prob = 0.5


# Loop through each simulation.
for i in range(n_simulations):
    # Loop through each step. 

    for j in range(1, n_steps):
        # Flip a coin.
        flip = np.random.rand()
        
        # Figure out how to step.
        if flip < step_prob:
            step = -1
        else:
            step = 1
            
        # Update our position.
        position[i, j] = position[i, j-1] + step
    
        
# Plot all of the trajectories together.
for i in range(n_simulations):
    # Remembering that `position` is just a two-dimensional matrix that is 
    # n_simulations by n_steps, we can get each step for a given simulation 
    # by indexing as position[i, :].
    plt.plot(steps, position[i, :], linewidth=1, alpha=1) 
    
# Add axis labels.
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()


# Compute the mean position at each step. 
mean_position = np.zeros(n_steps)
for i in range(n_steps):
    mean_position[i] = np.mean(position[:, i])

# Plot all of the simulations.
for i in range(n_simulations):
    plt.plot(steps, position[i, :], linewidth=1, alpha=1)
    
# Plot the mean as a thick red line. 
plt.plot(steps, mean_position, 'b-')

# Add the labels.
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()


# Hypothesis 2:
   
step_prob2 = 0.6
n_steps2 = 100 # Essentially number of steps.


    
# Set up a vector to store our positions.

position2 = np.zeros(n_steps2)

# Hypothesis 2

for i in range(1, n_steps2):
    # Flip a coin.
    flip2 = np.random.rand()
    
    # Figure out which way we should step.
    if flip2 < step_prob2:
        step2 = -1 # To the 'left'.
    else:
        step2 = 1# to the 'right'.
        
    # Update our position based on where we were in the last time point. 
    position2[i] = position2[i-1] + step2
    
# Number of step to the left in hypothesis 2:
nl2=int((n_steps-position2[i])/2)
print("The number of steps taken to the left in our second hypothesis is %s.Therefore, the number of steps taken to the right is %s." %(nl2, (100-nl2)))
steps2 = np.arange(0, n_steps2, 1)  # Arange from 0 to n_steps taking intervals of 1.

d2={'steps':np.array(steps2), 'position':np.array(position2)}
df2=pd.DataFrame(d2)

P2=np.log(((math.factorial(n_steps2))*((step_prob2)**(n_steps2-nl2))*((1-step_prob2)**(nl2)))/(((math.factorial(n_steps2-nl2)))*(math.factorial(nl2))))
print("The log likelihood for our second hypothesis is %s" %(P2))
#print(steps)
#print(position)
# Plot it!
plt.plot(steps2, position2)
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()

n_simulations2 = 1000

# Make a new position vector. This will include all simulations.
position2 = np.zeros((n_simulations, n_steps))


# Redefine our step probabilities just to be clear. 
step_prob2 = 0.6


# Loop through each simulation.
for i in range(n_simulations2):
    # Loop through each step. 

    for j in range(1, n_steps2):
        # Flip a coin.
        flip2 = np.random.rand()
        
        # Figure out how to step.
        if flip2 < step_prob2:
            step2 = -1
        else:
            step2 = 1
            
        # Update our position.
        position2[i, j] = position2[i, j-1] + step2
    
        
# Plot all of the trajectories together.
for i in range(n_simulations2):
    # Remembering that `position` is just a two-dimensional matrix that is 
    # n_simulations by n_steps, we can get each step for a given simulation 
    # by indexing as position[i, :].
    plt.plot(steps2, position2[i, :], linewidth=1, alpha=1) 
    
# Add axis labels.
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()


# Compute the mean position at eacNEh step. 
mean_position2 = np.zeros(n_steps2)
for i in range(n_steps2):
    mean_position2[i] = np.mean(position2[:, i])

# Plot all of the simulations.
for i in range(n_simulations2):
    plt.plot(steps2, position2[i, :], linewidth=1, alpha=1)
    
# Plot the mean as a thick red line. 
plt.plot(steps2, mean_position2, 'b-')

# Add the labels.
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.show()

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

LR=likelihood_ratio(P2, P)
p = chi2.sf(LR, 1)
print("The likehood ratio of our two hypotheses is %s" %(LR))
print('p-value: %.30f' % p)

# Behavior of likelihood with p

x=np.linspace(0.1, 1.0, num=10)
L=(((math.factorial(n_steps))*((x)**(n_steps-nl))*((1-x)**(nl)))/((math.factorial(nl))*(math.factorial(n_steps-nl))))
fig=plt.figure()
values=['0.1', '0.2', '0.3','0.4', '0.5', '0.6','0.7','0.8','0.9','1.0']

plt.plot(x,L, 'r')
plt.xlabel('Probability of stepping to the right')
plt.ylabel('Likelihood')
plt.xticks(x, values)
plt.show()
