# Multi-Armed Bandit Problem
This project implements some of the reinforcement learning (RL) methods covered in Sutton and Barto's [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/RLbook2020.pdf) for the multi-armed bandit problem, which is a special RL problem that only contains 1 state ([more info](https://www.geeksforgeeks.org/multi-armed-bandit-problem-in-reinforcement-learning/)). The main.py file contains a series of experiments comparing various action-value methods: 
- the sample average method vs. the exponential recency-weighted average method,
- the 0.1-greedy method vs, 0.01-greedy method, and
- 0-initialization vs. the optimal initial values method.


<br>

## Experiment Set-Up (skip if not interested)
For the stationary bandit problem, the true values were sampled from the standard normal. The rewards came from a normal distribution with the mean set to the true value and 
standard deviation set to 1. 

As for the non-stationary version, the true values were initialized to 0 but scattered a bit with each time step
(I added a normally distributed increment with mean 0 and standard deviation of 0.01 with each step). Otherwise, the rewards worked the same way as the stationary version.

For the experiment, I ran the two systems for 10,000 time steps, and took the average reward value at each time step over 1,000 independent runs. 

<br>

## Results
### Sample Average vs. Exponential Recency-Weighted Average
Set-up details: for both action-value methods, I used an epsilon-greedy method with epsilon=0.1 to encourage exploration to some extent. For the recency-weighted method, I used a constant step size of 0.1.

Results: I found that the two methods performed similarly on the stationary problem (as expected), but the 
recency-weighted (or "Constant Step Size") method performed significantly better on the non-stationary problem (also as expected): <br><br>
![image](https://github.com/user-attachments/assets/53de243a-8edd-4362-aa07-38f543d06d85)<br>
*Figure 1: Stationary - both methods have similar performance.*<br><br>
![image](https://github.com/user-attachments/assets/19bc42c6-22ac-40df-a939-6038d53a0337)<br>
*Figure 2: Non-stationary - recency-weighted method performs much better.*

<br>

### 0.1-Greedy vs. 0.01-Greedy (exploration vs. exploitation trade-off)
Set-up details: I compared a couple epsilon-greedy methods on the stationary problem (0.1 and 0.01) with the same set up as before. 

Results: As expected, the 0.1-greedy method performs better 
at first because it discovers the optimal actions earlier, but the 0.01-greedy method performs better over the long run because it chooses the optimal method more often the the 0.1-greedy method.
<br><br>
![image](https://github.com/user-attachments/assets/326d59d8-42ef-4f40-ab3c-c7e759bf54c5)<br>
*Figure 3: Comparing different epsilon-greedy methods to see how much exploration is optimal (for this particular scenario).*

### 0-initialization vs. Optimal Initial Values
Set-up details: For optimal initial values, I initialized value estimates at +5, and tested the two methods on the stationary problem (this is a requirement for the optimal initial values method). The 0-init method uses epislon=0.1, whereas the optimal initial value method is purely greedy (epsilon=0).

Results: As expected, the optimal initial values method performs worse at the very start because it explores more, but it quickly outperforms the 0-init method because it tends to find the optimal actions earlier and more frequently.
<br><br>
![image](https://github.com/user-attachments/assets/c6a647f1-01e7-4d5e-bfb6-b7767726e02a)
*Figure 4: Line chart showing how optimal initial values can help encourage exploration for stationary problems.*

