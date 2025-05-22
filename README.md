# Multi-Armed Bandit Problem
I wrote this script for an optional exercise in Sutton and Barto's [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/RLbook2020.pdf) (exercise 2.5). 
The script sets up two RL systems, one using the sample average method for updating value estimates and the other using the exponential recency-weighted average method, and compares
their performance on both a stationary and non-stationary 10-armed bandit problem. 

<br>

## Set-Up
For the stationary bandit problem, the true values were sampled from the standard normal. The rewards came from a normal distribution with the mean set to the true value and 
standard deviation set to 1. 

As for the non-stationary version, the true values were initialized to 0 but scattered a bit with each time step
(I added a normally distributed increment with mean 0 and standard deviation of 0.01 with each step). Otherwise, the rewards worked the same way as the stationary version.

For both action-value methods, I used an epsilon-greedy method with epsilon=0.1 to encourage exploration to some extent. 

For the recency-weighted method, I used a constant step size of 0.1.

For the experiment, I ran the two systems for 10,000 time steps, and took the average reward value at each time step over 1,000 independent runs. 

<br>

## Results
I found that the two methods performed similarly on the stationary problem (as expected), but the 
recency-weighted (or "Constant Step Size") method performed significantly better on the non-stationary problem (also as expected): <br><br>
![image](https://github.com/user-attachments/assets/53de243a-8edd-4362-aa07-38f543d06d85)<br>
*Figure 1: Stationary - both methods have similar performance.*<br><br>
![image](https://github.com/user-attachments/assets/19bc42c6-22ac-40df-a939-6038d53a0337)<br>
*Figure 2: Non-stationary - recency-weighted method performs much better.*

<br>

## Exploration vs. Exploitation Trade-Off
Just for fun, I also compared a couple epsilon-greedy methods on the stationary problem (0.1 and 0.01) with the same set up as before. As expected, the 0.1-greedy method performs better 
at first because it discovers the optimal actions earlier, but the 0.01-greedy method performs better over the long run because it chooses the optimal method more often the the 0.1-greedy method.
<br><br>
![image](https://github.com/user-attachments/assets/326d59d8-42ef-4f40-ab3c-c7e759bf54c5)<br>
*Figure 3: Comparing different epsilon-greedy methods to see how much exploration is optimal (for this particular scenario).*
