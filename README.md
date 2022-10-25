# Tutorial to use this repository
This repository contains some codes that I used to test the KalmanNet-VI algorithm to solve the regulator problem for linear discrete-time systems
1. Download all the files in one folder directory on your PC
2. Open your MATLAB apps (min. R2021a)
3. Set your workspace folder into the downloaded foldes from step (1)
4. Open file 'main.m' in MATLAB
5. Run the code<br />
If you would like to change some parameters about :<br />
the **disturbance characteristics**, you could change the code (lines 17-18)<br />
the **LSTM parameters**, you should open KalmanNet.m (m-file) and then change the number of hidden units, max epochs, initial learning rate, and etc.<br />
# Algorithm Description 
In this study, we compare three combination method to adapt the LQG scheme. First scenario, Kalman Filter was used to solve the state estimation process and Value Iteration algorithm was used to solve regulation problem. In the second scenario,  
