clear all;clc;close all
N = 100; 
Rww = 0.01;
Rvv = .1;
Q = .01;
R = .1;
init = .1*ones(4,100);
%% Case Study 1 : Cart-Pole  (u:LQR)
cartpole = sysmdl_cartpole(N);
dataset_cp = GenerateSeq(cartpole.sys,N,Rww,Rvv,Q,R);
delta_x_cp = [dataset_cp.x(:,1:end-1);dataset_cp.y];
target_cp = [dataset_cp.x_nw(:,1:end-1);dataset_cp.y_nw];
tic;
[KG_cp,net_cp] = KalmanNet(delta_x_cp,target_cp,cartpole.sys.A,cartpole.sys.C,N);
time_kn_cp = toc;
%%  Implement KG for state estimation process (u:LQR)
delta_y_cp = dataset_cp.y-dataset_cp.y_nw;
x_hat_net_nw_cp = (KG_cp*delta_y_cp')+dataset_cp.x_nw(:,1:end-1);
x_hat_net_cp = (KG_cp*delta_y_cp')+(dataset_cp.x(:,1:end-1)-dataset_cp.x_nw(:,1:end-1));
y_hat_net_cp = cartpole.sys.C*x_hat_net_cp;
%% Implement KF for state estimation process (u:LQR)
x_hat_kf_cp = dataset_cp.x_hat(:,1:end-1);
%% Case Study 2 : Distillate (u:LQR)
distillate = sysmdl_distillate(N);
dataset_dt = GenerateSeq(distillate.sys,N,Rww,Rvv,Q,R);
delta_x_dt = [dataset_dt.x(:,1:end-1);dataset_dt.y];
target_dt = [dataset_dt.x_nw(:,1:end-1);dataset_dt.y_nw];
tic;
[KG_dt,net_dt] = KalmanNet(delta_x_dt,target_dt,distillate.sys.A,distillate.sys.C,N);
time_kn_dt = toc;
%%  Implement KG for state estimation process (u:LQR)
delta_y_dt = dataset_dt.y-dataset_dt.y_nw;
x_hat_net_nw_dt = (KG_dt*delta_y_dt')+dataset_dt.x_nw(:,1:end-1);
x_hat_net_dt = (KG_dt*delta_y_dt')+(dataset_dt.x(:,1:end-1)-dataset_dt.x_nw(:,1:end-1));
y_hat_net_dt = distillate.sys.C*x_hat_net_dt;
%% Implement KF for state estimation process (u:LQR)
x_hat_kf_dt = dataset_dt.x_hat(:,1:end-1);
%% MSE (u:LQR)
mse_kn_1 = mse(x_hat_net_cp,dataset_cp.x(:,1:end-1))
mse_kf_1 = mse(x_hat_kf_cp,dataset_cp.x(:,1:end-1))
mse_kn_2 = mse(x_hat_net_dt,dataset_dt.x(:,1:end-1))
mse_kf_2 = mse(x_hat_kf_dt,dataset_dt.x(:,1:end-1))
%% Time Elapsed (u:LQR)
time_kn_cp = time_kn_cp 
time_kf_cp = dataset_cp.time
time_kn_dt = time_kn_dt
time_kf_dt = dataset_dt.time
