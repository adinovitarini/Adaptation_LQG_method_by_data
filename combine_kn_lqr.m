%% Second Scenario : KalmanNet and LQR
function [P,u,x_hat,y,J,K] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net)
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
u = abs(u_kn_lqr);
y = y_hat_kn_lqr;
x_hat = abs(x_hat_kn_lqr);
J = J_kn_lqr;
K = KK;
