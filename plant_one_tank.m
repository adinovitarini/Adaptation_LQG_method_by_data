clear all;clc
g = 9.8e-1;
lp = 1.5;
mp = 0.1;
mk = 1;
mt = 1.1;
a = g/(lp*(4/3 -mp/(mp+mk)));
A = [0 1 0 0;0 0 a 0;0 0 0 1;0 0 a 0];
b = -1/(lp*(4/3 -mp/(mp+mk)));
B = [0;1/mt;0;b];
C = [0 1 0 1];
D = 0;
sys = ss(A,B,C,D,0.01);
N = 100;
u = 0;
%%  Apply Disturbance 
w = wgn(N,1,1);
v = w;
%%  Open Loop System
x = .1*ones(4,1);
for i = 1:N
    x(:,i+1) = A*x(:,i)+B*u+w(i);
    y(i) = C*x(:,i)+v(i);
    x(:,i) = x(:,i+1);
end
%% Weight Matrices 
Rww = 0.01;
Rvv = 1;
Q = .01;
R = .1;
%%  LQG Conventional 
[~,L,~] = kalman(sys,Rww,Rvv);
[K,~,cl] = lqr(sys,Q,R);
%%  Observer via KalmanNet
x_nw = ones(size(A,1),1);
for i = 1:N
    x_nw(:,i+1) = A*x_nw(:,i);
    y_nw(i) = C*x_nw(:,i);
    x_nw(:,i) = x_nw(:,i+1);
end
x_nw = x_nw(:,1:N);
delta_x = [x(:,2:end);y];
target = [x_nw;y_nw];
KG = KalmanNet(delta_x,target,A,C);
%%  Implement KG for state estimation process 
delta_y = y+(-1*y_nw);
x_hat_net = (KG*delta_y')+x_nw;
y_hat_net = C*x_hat_net;

%% First scenario : LQG konvensional 
x_lqg = x;
x_hat_lqg = x;
y_lqg = y;
L_lqg = L;
K_lqg = K;
for i = 1:N
    u_lqg(i) = -K_lqg*x_lqg(:,i);
    x_hat_lqg(:,i+1) = A*x_hat_lqg(:,i)+B*u_lqg(i);
    y_hat_lqg(i) = C*x_hat_lqg(:,i);
    x_lqg(:,i+1) = (A-B*K_lqg)*x_lqg(:,i)+L_lqg*(y_hat_lqg(i)-y_lqg(i));
    y_lqg(i) = C*x_lqg(:,i);
end
%%  Second scenario : KF-VI   
[P_kf_vi,u_kf_vi,x_hat_kf_vi,y_kf_vi,J_kf_vi,K_kf_vi,L_kf_vi] = combine_kf_vi(A,B,C,Q,R,N,L,y);
%%  Third scenario : KalmanNet-LQR   
[P_kn_lqr,u_kn_lqr,x_hat_kn_lqr,y_kn_lqr,J_kn_lqr,K_kn_lqr] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net);
%%  Fourth scenario : KalmanNet-VI
[P_kn_vi,u_kn_vi,x_hat_kn_vi,y_kn_vi,J_kn_vi,K_kn_vi] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net);
%% Stability Analysis 
for i = 1:N
eig_kf_vi(:,i) = eig(A-B*K_kf_vi(i,:)-L_kf_vi*C);
eig_kn_lqr(:,i) = eig(A-B*K_kn_lqr-KG(1,i)*C);
eig_kn_vi(:,i) = eig(A-B*K_kn_vi(i,:)-KG(1,i)*C);
end
norm_eig_kf_vi = zeros(size(A,1),1);
norm_eig_kn_lqr = norm_eig_kf_vi;
norm_eig_kn_vi = norm_eig_kn_lqr;
for j = 1:size(A,1)
    norm_eig_kf_vi(j,1) = max(eig_kf_vi(j,:));
    norm_eig_kn_lqr(j,1) = max(eig_kn_lqr(j,:));
    norm_eig_kn_vi(j,1) = max(eig_kn_vi(j,:));
end
%% Display the output 
clc
fprintf('Kalman gain from KF : %f \n',norm(L))
fprintf('Kalman gain from KN : %f \n',norm(KG))
fprintf('Controller gain from LQR : %f \n',norm(K))
fprintf('Controller gain from VI : %f \n',norm(K_kf_vi))
fprintf('Controller gain from LQR : %f \n',norm(K_kn_lqr))
fprintf('Controller gain from VI : %f \n',norm(K_kn_vi))
