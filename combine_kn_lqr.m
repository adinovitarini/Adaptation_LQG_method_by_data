%% Second Scenario : KalmanNet and LQR
function [kn_lqr] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net)
tic
[P,KK,~] = idare(A,B,Q,R);
x_hat_kn_lqr = x_hat_net;
for i = 1:N
    u_kn_lqr(1,i) = -KK*x_hat_kn_lqr(:,i);
    y_hat_kn_lqr(i) = C*x_hat_kn_lqr(:,i);
    x_hat_kn_lqr(:,i+1) = A*x_hat_kn_lqr(:,i)+B*u_kn_lqr(:,i);
end
% Value function 
J_kn_lqr = value_func(x_hat_kn_lqr,u_kn_lqr,Q,R,N);
% % output 
u = (u_kn_lqr);
y = y_hat_kn_lqr;
x_hat = abs(x_hat_kn_lqr);
J = J_kn_lqr;
K = KK;
time_elapsed = toc;
kn_lqr.P = P;
kn_lqr.u = u;
kn_lqr.x_hat = x_hat;
kn_lqr.y = y;
kn_lqr.J = J;
kn_lqr.K = K;
kn_lqr.time = time_elapsed;